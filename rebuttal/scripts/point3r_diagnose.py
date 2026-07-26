#!/usr/bin/env python3
"""
E5 step 1 — Does the DDD3R diagnosis transfer to a DIFFERENT recurrent 3D backbone?

Target: Point3R (NeurIPS 2025), an explicit *spatial pointer memory* model built on the
DUSt3R ViT-L backbone — architecturally unrelated to CUT3R's implicit 768-token state.

Point3R's memory merge (src/dust3r/point3r.py, _forward_addmemory_merge) does:

    feat_avg = feat_sum / count                  # mean of the NEW features only
    memory_feat_j[unique_indices] = feat_avg     # hard overwrite of an existing pointer

i.e. an existing memory entry is *fully replaced* by new content: beta = 1, no dampening.
That is exactly the extreme of failure mode M1 in our paper.

This script is READ-ONLY. It monkey-patches the merge to record, for every merge event,
    delta = feat_avg - memory_feat[idx]
and reports:
  * M1: relative update magnitude   ||delta|| / ||memory_feat[idx]||
  * M3: drift energy cos^2(delta_t, delta_{t-1}) per pointer

Comparable to our Table 3 (TUM 0.398 +/- 0.041, ScanNet 0.598 +/- 0.054).

Usage:
  CUDA_VISIBLE_DEVICES=1 python rebuttal/scripts/point3r_diagnose.py \
      --ckpt rebuttal/external/Point3R_ckpt/point3r.pth \
      --seq_dir <dir of images> --n_frames 100
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P3R = os.path.join(WT, "external", "Point3R")
sys.path.insert(0, P3R)
sys.path.insert(0, os.path.join(P3R, "src"))

STATS = {"rel_mag": [], "cos": [], "n_merge": [], "n_add": [], "mem_size": []}
_PREV = {}  # pointer index -> previous delta (numpy)


def patch(model):
    """Wrap the merge so we observe delta without changing behaviour."""
    import dust3r.point3r as P
    orig = P.Point3R._forward_addmemory_merge

    def wrapped(self, i, pts3d, init_memory_feat, memory_feat, memory_pos,
                feat_i, dec_i, shape_i):
        def as_t(x):
            """memory_feat may be a tensor [B,N,C] or a per-batch list; take batch 0."""
            if x is None:
                return None
            if isinstance(x, (list, tuple)):
                return None if len(x) == 0 else x[0].detach().clone().float()
            return x[0].detach().clone().float()

        before = as_t(memory_feat)
        out = orig(self, i, pts3d, init_memory_feat, memory_feat, memory_pos,
                   feat_i, dec_i, shape_i)
        after = as_t(out[0])
        if before is not None and after is not None and before.shape[0] > 0:
            n = min(before.shape[0], after.shape[0])
            b = before[:n]
            a = after[:n]
            d = a - b                                    # [n, C]
            nrm = d.norm(dim=-1)
            changed = nrm > 1e-8                         # entries actually overwritten
            if changed.any():
                idxs = torch.nonzero(changed).squeeze(-1)
                rel = (nrm[idxs] / b[idxs].norm(dim=-1).clamp(min=1e-8))
                STATS["rel_mag"].extend(rel.cpu().numpy().tolist())
                dn = torch.nn.functional.normalize(d[idxs], dim=-1).cpu().numpy()
                for k, ptr in enumerate(idxs.cpu().numpy().tolist()):
                    p = _PREV.get(ptr)
                    if p is not None:
                        STATS["cos"].append(float(np.dot(p, dn[k])))
                    _PREV[ptr] = dn[k]
                STATS["n_merge"].append(int(changed.sum()))
            STATS["mem_size"].append(int(after.shape[0]))
            STATS["n_add"].append(int(after.shape[0] - before.shape[0]))
        return out

    P.Point3R._forward_addmemory_merge = wrapped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seq_dir", required=True)
    ap.add_argument("--n_frames", type=int, default=100)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from dust3r.point3r import Point3R, Point3RConfig  # noqa: F401
    from dust3r.utils.image import load_images

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # The released checkpoint stores args as {'model': '<config expr>'}, while the repo's
    # load_model() expects an argparse Namespace (ckpt['args'].model). Build directly.
    # weights_only=True: this checkpoint holds only tensors plus a {'model': str} dict.
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    cfg = ck["args"]["model"] if isinstance(ck["args"], dict) else ck["args"].model
    cfg = cfg.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
    # The upstream repo evaluates this config string directly. We keep that mechanism but
    # constrain it: it must be exactly a Point3R(Point3RConfig(...)) constructor call, with
    # no attribute access, indexing, or statement separators that could smuggle in code.
    if not (cfg.startswith("Point3R(Point3RConfig(") and cfg.endswith(")")):
        raise ValueError(f"unexpected model config expression: {cfg[:80]}")
    if any(tok in cfg for tok in (";", "import", "__", "lambda", "\n", "exec", "open(")):
        raise ValueError("model config expression contains disallowed tokens")
    model = eval(cfg, {"Point3R": Point3R, "Point3RConfig": Point3RConfig,
                       "inf": float("inf"), "nan": float("nan"),
                       "__builtins__": {}})  # noqa: S307 - validated literal above
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    print(f"loaded: {len(missing)} missing, {len(unexpected)} unexpected keys")
    model = model.to(dev).eval()
    patch(model)

    exts = (".png", ".jpg", ".jpeg", ".JPG", ".PNG")
    paths = sorted(os.path.join(args.seq_dir, f)
                   for f in os.listdir(args.seq_dir) if f.endswith(exts))[:args.n_frames]
    print(f"{len(paths)} frames from {args.seq_dir}")
    imgs = load_images(paths, size=args.size)

    views = []
    for k, d in enumerate(imgs):
        views.append({
            "img": d["img"],
            "ray_map": torch.full((d["img"].shape[0], 6, d["img"].shape[-2],
                                   d["img"].shape[-1]), torch.nan),
            "true_shape": torch.from_numpy(d["true_shape"]),
            "idx": k, "instance": str(k),
            "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(0),
            "img_mask": torch.tensor(True).unsqueeze(0),
            "ray_mask": torch.tensor(False).unsqueeze(0),
            "update": torch.tensor(True).unsqueeze(0),
            "reset": torch.tensor(False).unsqueeze(0),
        })
    for v in views:
        for k2 in ("img", "ray_map", "true_shape", "camera_pose",
                   "img_mask", "ray_mask", "update", "reset"):
            v[k2] = v[k2].to(dev)

    with torch.no_grad():
        # Use Point3R's own entry point: the merge path requires point3r_tag=True.
        model._forward_merge(views, point3r_tag=True)

    rel = np.array(STATS["rel_mag"]); cos = np.array(STATS["cos"])
    print("\n" + "=" * 66)
    print(f"Point3R memory-merge diagnosis  ({len(rel)} overwrite events)")
    print("=" * 66)
    if rel.size:
        print(f"  M1 relative update magnitude ||d||/||mem||: "
              f"mean {rel.mean():.3f}  median {np.median(rel):.3f}")
    if cos.size:
        de = cos ** 2
        print(f"  M3 cos(delta_t, delta_t-1): {cos.mean():.3f} +/- {cos.std():.3f}")
        print(f"  M3 drift energy (cos^2)   : {de.mean():.3f} +/- {de.std():.3f}   "
              f"(n={de.size})")
        print(f"     reference — TUM 0.398 +/- 0.041 | ScanNet 0.598 +/- 0.054")
        v = de.mean()
        print(f"  => drift energy is {'HIGH (>0.5, ScanNet-like)' if v > 0.5 else 'MODERATE/LOW'}"
              f"; directional redundancy {'DOES' if v > 0.3 else 'does NOT'} transfer")
    else:
        print("  no repeated overwrites of the same pointer — M3 not measurable")
    if STATS["mem_size"]:
        print(f"  memory grows {STATS['mem_size'][0]} -> {STATS['mem_size'][-1]} pointers")

    out = args.out or os.path.join(WT, "results", "point3r_diagnosis.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({
        "n_events": int(rel.size),
        "rel_mag_mean": float(rel.mean()) if rel.size else None,
        "rel_mag_median": float(np.median(rel)) if rel.size else None,
        "cos_mean": float(cos.mean()) if cos.size else None,
        "cos_std": float(cos.std()) if cos.size else None,
        "drift_energy_mean": float((cos ** 2).mean()) if cos.size else None,
        "drift_energy_std": float((cos ** 2).std()) if cos.size else None,
        "mem_size_first": STATS["mem_size"][0] if STATS["mem_size"] else None,
        "mem_size_last": STATS["mem_size"][-1] if STATS["mem_size"] else None,
        "seq_dir": args.seq_dir, "n_frames": len(paths),
    }, open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
