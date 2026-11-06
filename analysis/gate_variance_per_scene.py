"""Per-scene Var(beta_t) vs TTT3R-vs-CUT3R improvement.

Direct test of "TTT3R's adaptive gate carries no actionable timing
signal" (analysis 3.2 punchline). Earlier we correlated
Var(cos(delta_t, delta_{t-1})) -- the cross-attention input -- with
brake-vs-random improvement, which a reviewer reasonably called
indirect. Here we measure the actual frame-averaged TTT3R gate
beta_t over the full sequence per scene, take its temporal variance,
and Pearson-correlate it with the per-scene improvement of TTT3R
over the un-gated CUT3R baseline.

Output:
  analysis_results/a2b_direct_gate_variance/<dataset>_data.npz
  analysis_results/a2b_direct_gate_variance/<dataset>.log

Usage:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python \
      analysis/gate_variance_per_scene.py [--limit N] [--dataset scannet|tum]
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

BASE = Path("/home/szy/research/TTT3R")
sys.path.insert(0, str(BASE / "src"))
sys.path.insert(0, str(BASE / "analysis"))

# Silence verbose per-frame prints from dust3r.utils.image.load_images
import dust3r.utils.image as _img_mod
_orig_load_images = _img_mod.load_images
def _quiet_load_images(folder_or_list, size, square_ok=False, verbose=False):
    return _orig_load_images(folder_or_list, size, square_ok=square_ok, verbose=False)
_img_mod.load_images = _quiet_load_images

from a1a2_gate_dynamics import (
    SIZE, build_views, get_scene_images, run_analysis,
)
# Re-bind the load_images symbol inside a1a2_gate_dynamics if it imported it directly
import a1a2_gate_dynamics as _agd
_agd_load_images_attr = getattr(_agd, "load_images", None)
if _agd_load_images_attr is _orig_load_images:
    _agd.load_images = _quiet_load_images

from dust3r.model import ARCroco3DStereo

OUT_DIR = BASE / "analysis_results" / "a2b_direct_gate_variance"


def parse_per_scene_from_logs(method, dataset_dir):
    log_path = dataset_dir / method / "_error_log.txt"
    if not log_path.exists():
        return {}
    out = {}
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Average"):
                continue
            try:
                float(line)
                continue
            except ValueError:
                pass
            m = re.match(r"^[\w_]+-(.+?)\s*\|\s*ATE:\s*([\d.]+)", line)
            if m:
                out[m.group(1).strip()] = float(m.group(2))
    return out


def parse_per_scene_metric_dir(method_dir):
    out = {}
    if not method_dir.exists():
        return out
    for f in method_dir.iterdir():
        if not f.name.endswith("_eval_metric.txt"):
            continue
        scene = f.name.replace("_eval_metric.txt", "")
        try:
            with open(f) as fh:
                txt = fh.read()
            m = re.search(r"APE w\.r\.t\. translation.*?mean\s+([\d.]+)", txt, re.DOTALL)
            if m:
                out[scene] = float(m.group(1))
        except (OSError, ValueError):
            continue
    return out


def collect_ates(method, dataset_dir):
    res = parse_per_scene_from_logs(method, dataset_dir)
    if res:
        return res
    return parse_per_scene_metric_dir(dataset_dir / method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="scannet", choices=["scannet", "tum"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", action="store_true",
                    help="Skip scenes already in the output npz")
    ap.add_argument("--shard", type=str, default="",
                    help="Shard spec 'I/N' to process only scenes with "
                         "(global_index %% N) == I; lets multiple GPUs split work.")
    ap.add_argument("--tag", type=str, default="",
                    help="Suffix appended to output npz filename, e.g. '_g0'.")
    args = ap.parse_args()

    if args.dataset == "scannet":
        scan_dir = BASE / "eval_results/relpose/scannet_s3_1000"
    else:
        scan_dir = BASE / "eval_results/relpose/tum_s1_1000"

    cut3r_ates = collect_ates("cut3r", scan_dir)
    ttt3r_ates = collect_ates("ttt3r", scan_dir)
    common_full = sorted(set(cut3r_ates) & set(ttt3r_ates))
    print(f"[{args.dataset}] {len(common_full)} scenes have both CUT3R and TTT3R ATE", flush=True)

    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        common = [s for k, s in enumerate(common_full) if k % n == i]
        print(f"  shard {args.shard}: {len(common)} of {len(common_full)} scenes", flush=True)
    else:
        common = common_full

    if args.limit:
        common = common[: args.limit]
        print(f"  limiting to first {len(common)}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = args.tag if args.tag else ""
    out_path = OUT_DIR / f"{args.dataset}{suffix}_data.npz"

    # Resume support
    done = set()
    scenes, gate_vars, ate_cs, ate_ts, imps = [], [], [], [], []
    if args.resume and out_path.exists():
        old = np.load(out_path, allow_pickle=True)
        for s, gv, c, t, ip in zip(old["scene"], old["gate_var"],
                                    old["ate_cut3r"], old["ate_ttt3r"],
                                    old["improvement"]):
            scenes.append(str(s))
            gate_vars.append(float(gv))
            ate_cs.append(float(c))
            ate_ts.append(float(t))
            imps.append(float(ip))
            done.add(str(s))
        print(f"  resuming: {len(done)} scenes already done", flush=True)

    print("Loading model...", flush=True)
    weights = BASE / "model" / "cut3r_512_dpt_4_64.pth"
    model = ARCroco3DStereo.from_pretrained(str(weights)).to(args.device).eval()

    for i, scene in enumerate(common):
        if scene in done:
            continue
        img_paths = get_scene_images(args.dataset, scene)
        if not img_paths:
            print(f"  [{i+1}/{len(common)}] SKIP {scene}: no images", flush=True)
            continue
        n_frames = len(img_paths)

        model.config.model_update_type = "ttt3r"
        try:
            data = run_analysis(model, img_paths, args.device)
        except Exception as e:
            print(f"  [{i+1}/{len(common)}] FAIL {scene}: {e}", flush=True)
            continue
        gates = np.array(data.get("gate_history", []))
        if gates.size < 4:
            print(f"  [{i+1}/{len(common)}] SKIP {scene}: gate history too short", flush=True)
            continue
        gate_var = float(np.var(gates))
        c, t = cut3r_ates[scene], ttt3r_ates[scene]
        improvement = (c - t) / c * 100.0 if c > 1e-6 else float("nan")

        scenes.append(scene)
        gate_vars.append(gate_var)
        ate_cs.append(c)
        ate_ts.append(t)
        imps.append(improvement)

        print(f"  [{i+1}/{len(common)}] {scene}: T={n_frames}, "
              f"Var(beta)={gate_var:.5f}, ATE c={c:.3f} t={t:.3f}, "
              f"improvement={improvement:+.1f}%", flush=True)

        # Save after every scene so progress is durable
        np.savez(out_path,
                 scene=np.array(scenes),
                 gate_var=np.array(gate_vars),
                 ate_cut3r=np.array(ate_cs),
                 ate_ttt3r=np.array(ate_ts),
                 improvement=np.array(imps),
                 pearson_r=np.nan, pearson_p=np.nan)

    if len(gate_vars) >= 3:
        gv = np.array(gate_vars)
        ip = np.array(imps)
        valid = ~(np.isnan(gv) | np.isnan(ip))
        r, p = stats.pearsonr(gv[valid], ip[valid])
    else:
        r, p = float("nan"), float("nan")

    np.savez(out_path,
             scene=np.array(scenes),
             gate_var=np.array(gate_vars),
             ate_cut3r=np.array(ate_cs),
             ate_ttt3r=np.array(ate_ts),
             improvement=np.array(imps),
             pearson_r=r, pearson_p=p)
    print(f"\nSaved {out_path}", flush=True)
    print(f"n={len(gate_vars)}, Pearson r={r:+.3f}, p={p:.3f}", flush=True)


if __name__ == "__main__":
    main()
