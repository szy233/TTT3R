#!/usr/bin/env python3
"""
E4 step 1 — measure drift energy on ARKitScenes.

MUST run before any ARKitScenes ATE. The measured value selects which branch of the
pre-registered conditional P2 applies (see rebuttal/docs/E4_arkitscenes_PREREGISTRATION.md).

Reuses analysis/a4_delta_direction.py's model loading and delta analysis verbatim, so the
number is directly comparable to the paper's Table 3 (TUM 0.398, ScanNet 0.598).

Usage:
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:analysis \
  python rebuttal/scripts/arkit_drift_energy.py
"""
import json
import os
import sys
from glob import glob
from pathlib import Path

import numpy as np
import torch

BASE = Path("/home/szy/research/TTT3R")
WORKTREE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "analysis"))
sys.path.insert(0, str(WORKTREE / "src"))

from a4_delta_direction import load_model, run_detailed_delta_analysis  # noqa: E402

ARKIT = WORKTREE / "data/long_arkit_s1"
N_FRAMES = 500          # per Amendment 1
OUT = WORKTREE / "rebuttal/results/arkit_drift_energy.json"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device)

    scenes = sorted(d.name for d in ARKIT.iterdir()
                    if d.is_dir() and (d / f"rgb_{N_FRAMES}").is_dir())
    print(f"ARKitScenes: {len(scenes)} scenes with {N_FRAMES}f\n")

    stats = []
    for i, s in enumerate(scenes):
        imgs = sorted(glob(str(ARKIT / s / f"rgb_{N_FRAMES}" / "*.png")))[:N_FRAMES]
        if not imgs:
            print(f"  [{i+1}/{len(scenes)}] SKIP {s}")
            continue
        print(f"  [{i+1}/{len(scenes)}] {s} ({len(imgs)} frames)...", end=" ", flush=True)
        data = run_detailed_delta_analysis(model, imgs, device)
        cos = data["cosines"][1:]
        de = data["drift_energy"]
        rec = {
            "scene": s,
            "n_frames": len(imgs),
            "cos_mean": float(np.mean(cos)),
            "cos_std": float(np.std(cos)),
            "drift_energy_mean": float(np.mean(de)),
            "drift_energy_std": float(np.std(de)),
        }
        stats.append(rec)
        print(f"cos={rec['cos_mean']:.3f}, drift_e={rec['drift_energy_mean']:.3f}",
              flush=True)

    de_means = [s["drift_energy_mean"] for s in stats]
    cos_means = [s["cos_mean"] for s in stats]
    summary = {
        "dataset": "arkitscenes",
        "n_scenes": len(stats),
        "n_frames": N_FRAMES,
        "drift_energy_mean": float(np.mean(de_means)),
        "drift_energy_std": float(np.std(de_means)),
        "cos_mean": float(np.mean(cos_means)),
        "cos_std": float(np.std(cos_means)),
        "per_scene": stats,
    }

    print("\n" + "=" * 62)
    print(f"ARKitScenes drift energy: {summary['drift_energy_mean']:.3f} "
          f"± {summary['drift_energy_std']:.3f}  (n={len(stats)} scenes)")
    print(f"                     cos: {summary['cos_mean']:.3f} "
          f"± {summary['cos_std']:.3f}")
    print("-" * 62)
    print("reference (paper Table 3):  TUM 0.398 ± 0.041 | ScanNet 0.598 ± 0.054")
    e = summary["drift_energy_mean"]
    print("-" * 62)
    print(f"P1 (pre-registered: drift energy > 0.50): "
          f"{'HOLDS' if e > 0.50 else 'FAILS'}  (measured {e:.3f})")
    if e > 0.50:
        print("P2 branch selected -> ScanNet-like: predict brake < const < ortho")
    elif e < 0.45:
        print("P2 branch selected -> TUM-like: predict ortho < brake < const")
    else:
        print(f"P2: measured {e:.3f} falls in the ambiguous gap [0.45, 0.50] — "
              "record as indeterminate, no ranking predicted")
    print("=" * 62)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
