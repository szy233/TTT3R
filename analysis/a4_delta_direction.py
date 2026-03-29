"""
A4: Delta Direction Analysis — cos(δ_t, δ_{t-1}) distribution across datasets.
Compare ScanNet vs TUM to understand why ortho fails on ScanNet.

Key metrics per scene:
- cos mean / std / distribution
- cos temporal stability (how much drift_dir changes)
- drift energy ratio (% of delta in drift direction)

Usage:
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python analysis/a4_delta_direction.py
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob

BASE = Path("/home/szy/research/TTT3R")
OUT_DIR = BASE / "analysis_results/a4_delta_direction"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = str(BASE / "model/cut3r_512_dpt_4_64.pth")
SIZE = 512
MAX_FRAMES = 200  # cap to match eval pipeline


def load_model(device="cuda"):
    from dust3r.model import ARCroco3DStereo
    model = ARCroco3DStereo.from_pretrained(MODEL_PATH)
    model = model.to(device)
    model.eval()
    model.config.model_update_type = "ttt3r"
    return model


def build_views(img_paths, size):
    from dust3r.utils.image import load_images
    images = load_images(img_paths, size=size)
    views = []
    for i, img_dict in enumerate(images):
        view = {
            "img": img_dict["img"],
            "ray_map": torch.full(
                (img_dict["img"].shape[0], 6,
                 img_dict["img"].shape[-2], img_dict["img"].shape[-1]),
                torch.nan),
            "true_shape": torch.from_numpy(img_dict["true_shape"]),
            "idx": i,
            "instance": str(i),
            "camera_pose": torch.from_numpy(
                np.eye(4, dtype=np.float32)).unsqueeze(0),
            "img_mask": torch.tensor(True).unsqueeze(0),
            "ray_mask": torch.tensor(False).unsqueeze(0),
            "update": torch.tensor(True).unsqueeze(0),
            "reset": torch.tensor(False).unsqueeze(0),
        }
        views.append(view)
    return views


def get_scene_images(dataset, scene_name):
    if dataset == "scannet":
        img_dir = BASE / f"data/long_scannet_s3/{scene_name}/color_1000"
    elif dataset == "tum":
        img_dir = BASE / f"data/long_tum_s1/{scene_name}/rgb_1000"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    if not img_dir.exists():
        return []
    return sorted(glob(str(img_dir / "*")))


def run_delta_analysis(model, img_paths, device="cuda"):
    """Run model and collect per-frame, per-token delta cosine statistics."""
    views = build_views(img_paths, SIZE)
    with torch.no_grad():
        ress, analysis_data = model.forward_recurrent_analysis(views, device=device)

    cosines = np.array(analysis_data['cosine_history'])
    delta_norms = np.array(analysis_data['delta_norm_history'])
    return cosines, delta_norms


def run_detailed_delta_analysis(model, img_paths, device="cuda", beta=0.95):
    """
    Run model and compute drift energy ratio using EMA drift direction.
    This simulates what ttt3r_ortho does internally.
    """
    from einops import rearrange
    from dust3r.utils.image import load_images

    views = build_views(img_paths, SIZE)

    # Move views to device
    for v in views:
        for k in v:
            if isinstance(v[k], torch.Tensor):
                v[k] = v[k].to(device)

    model.config.model_update_type = "ttt3r"

    # We'll hook into forward_recurrent_analysis to get raw deltas
    # Instead, use the analysis mode which already returns cosine history
    with torch.no_grad():
        ress, analysis_data = model.forward_recurrent_analysis(views, device=device)

    cosines = np.array(analysis_data['cosine_history'])  # frame-level mean cos
    delta_norms = np.array(analysis_data['delta_norm_history'])

    # Compute drift energy ratio from cosine values
    # If cos(δ_t, drift_dir) ≈ c, then drift_energy ≈ c^2 (fraction in drift direction)
    # But we only have cos(δ_t, δ_{t-1}), not cos(δ_t, EMA_drift)
    # Approximate: with β=0.95, drift_dir ≈ recent direction, so cos(δ,δ_prev) is a proxy

    # Drift energy ratio = cos^2 (projection fraction)
    drift_energy = cosines[1:] ** 2  # skip frame 0

    return {
        'cosines': cosines,
        'delta_norms': delta_norms,
        'drift_energy': drift_energy,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device)

    # Get all scenes
    scannet_scenes = sorted([
        d.name for d in (BASE / "data/long_scannet_s3").iterdir()
        if d.is_dir() and d.name.startswith("scene")
    ])
    tum_scenes = sorted([
        d.name for d in (BASE / "data/long_tum_s1").iterdir()
        if d.is_dir() and d.name.startswith("rgbd")
    ])

    print(f"ScanNet: {len(scannet_scenes)} scenes", flush=True)
    print(f"TUM: {len(tum_scenes)} scenes", flush=True)

    all_results = {}

    for dataset, scenes in [("tum", tum_scenes), ("scannet", scannet_scenes)]:
        print(f"\n{'='*60}")
        print(f"  Dataset: {dataset} ({len(scenes)} scenes)")
        print(f"{'='*60}")

        dataset_cosines_all = []  # all cosine values across scenes
        scene_stats = []

        for si, scene in enumerate(scenes):
            img_paths = get_scene_images(dataset, scene)
            if not img_paths:
                print(f"  [{si+1}/{len(scenes)}] SKIP {scene}: no images")
                continue

            img_paths = img_paths[:MAX_FRAMES]
            print(f"  [{si+1}/{len(scenes)}] {scene} ({len(img_paths)} frames)...", end=" ", flush=True)

            data = run_detailed_delta_analysis(model, img_paths, device)
            cos = data['cosines'][1:]  # skip frame 0
            drift_e = data['drift_energy']

            stats = {
                'scene': scene,
                'n_frames': len(img_paths),
                'cos_mean': float(np.mean(cos)),
                'cos_std': float(np.std(cos)),
                'cos_median': float(np.median(cos)),
                'cos_min': float(np.min(cos)),
                'cos_max': float(np.max(cos)),
                'cos_q25': float(np.percentile(cos, 25)),
                'cos_q75': float(np.percentile(cos, 75)),
                'pct_negative': float(np.mean(cos < 0) * 100),
                'drift_energy_mean': float(np.mean(drift_e)),
                'drift_energy_std': float(np.std(drift_e)),
                'delta_norm_mean': float(np.mean(data['delta_norms'])),
            }
            scene_stats.append(stats)
            dataset_cosines_all.extend(cos.tolist())

            print(f"cos={stats['cos_mean']:.3f}±{stats['cos_std']:.3f}, "
                  f"drift_e={stats['drift_energy_mean']:.3f}, "
                  f"neg={stats['pct_negative']:.1f}%", flush=True)

        all_results[dataset] = {
            'scene_stats': scene_stats,
            'all_cosines': np.array(dataset_cosines_all),
        }

        # Dataset-level summary
        if scene_stats:
            cos_means = [s['cos_mean'] for s in scene_stats]
            drift_means = [s['drift_energy_mean'] for s in scene_stats]
            pct_negs = [s['pct_negative'] for s in scene_stats]
            print(f"\n  --- {dataset} Summary ---")
            print(f"  cos mean:        {np.mean(cos_means):.4f} ± {np.std(cos_means):.4f}")
            print(f"  cos median:      {np.median(cos_means):.4f}")
            print(f"  drift energy:    {np.mean(drift_means):.4f} ± {np.std(drift_means):.4f}")
            print(f"  % negative cos:  {np.mean(pct_negs):.1f}% ± {np.std(pct_negs):.1f}%")

    # ── Comparison Plot ────────────────────────────────────────────────
    print("\n=== Generating comparison plots ===")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Histogram of cos values
    ax = axes[0]
    for dataset, color, label in [("tum", "#4477AA", "TUM"), ("scannet", "#EE7733", "ScanNet")]:
        if dataset in all_results:
            ax.hist(all_results[dataset]['all_cosines'], bins=100, alpha=0.6,
                    color=color, label=label, density=True)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("cos(δ_t, δ_{t-1})")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Frame-Level Cosine Similarity")
    ax.legend()
    ax.grid(True, alpha=0.2)

    # Plot 2: Per-scene cos mean distribution
    ax = axes[1]
    for dataset, color, label in [("tum", "#4477AA", "TUM"), ("scannet", "#EE7733", "ScanNet")]:
        if dataset in all_results:
            means = [s['cos_mean'] for s in all_results[dataset]['scene_stats']]
            ax.hist(means, bins=30, alpha=0.6, color=color, label=label, density=True)
    ax.set_xlabel("Per-Scene Mean cos(δ_t, δ_{t-1})")
    ax.set_ylabel("Density")
    ax.set_title("Per-Scene Cosine Mean Distribution")
    ax.legend()
    ax.grid(True, alpha=0.2)

    # Plot 3: Per-scene drift energy
    ax = axes[2]
    for dataset, color, label in [("tum", "#4477AA", "TUM"), ("scannet", "#EE7733", "ScanNet")]:
        if dataset in all_results:
            drift_e = [s['drift_energy_mean'] for s in all_results[dataset]['scene_stats']]
            ax.hist(drift_e, bins=30, alpha=0.6, color=color, label=label, density=True)
    ax.set_xlabel("Per-Scene Mean Drift Energy (cos²)")
    ax.set_ylabel("Density")
    ax.set_title("Per-Scene Drift Energy Distribution")
    ax.legend()
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(str(OUT_DIR / "a4_scannet_vs_tum_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / 'a4_scannet_vs_tum_comparison.png'}")

    # Plot 4: Per-scene cos_mean vs ortho ATE improvement
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    import re

    for dataset, eval_dir_name, color, marker in [
        ("scannet", "scannet_s3_1000", "#EE7733", "o"),
        ("tum", "tum_s1_1000", "#4477AA", "s"),
    ]:
        if dataset not in all_results:
            continue

        # Parse ortho and random ATE
        eval_dir = BASE / f"eval_results/relpose/{eval_dir_name}"
        ortho_config = "ttt3r_ortho_an05_ad005" if dataset == "scannet" else "ttt3r_ortho_an05_ad005"
        random_config = "ttt3r_random"

        ortho_ates = {}
        random_ates = {}

        for config, ates_dict in [(ortho_config, ortho_ates), (random_config, random_ates)]:
            log_path = eval_dir / config / "_error_log.txt"
            if log_path.exists():
                with open(log_path) as f:
                    for line in f:
                        m = re.match(r"^[\w_]+-(.+?)\s*\|\s*ATE:\s*([\d.]+)", line.strip())
                        if m:
                            ates_dict[m.group(1).strip()] = float(m.group(2))

        # Match scenes
        x_cos = []
        y_improv = []
        for s in all_results[dataset]['scene_stats']:
            scene = s['scene']
            if scene in ortho_ates and scene in random_ates:
                improv = (random_ates[scene] - ortho_ates[scene]) / random_ates[scene] * 100
                x_cos.append(s['cos_mean'])
                y_improv.append(improv)

        if x_cos:
            ax.scatter(x_cos, y_improv, c=color, marker=marker, s=40, alpha=0.7,
                      edgecolors="k", linewidths=0.3, label=f"{dataset} ({len(x_cos)} scenes)")

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Per-Scene Mean cos(δ_t, δ_{t-1})")
    ax.set_ylabel("Ortho ATE Improvement vs Random (%)")
    ax.set_title("Delta Direction Consistency vs Ortho Effectiveness")
    ax.legend()
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(str(OUT_DIR / "a4_cos_vs_ortho_improvement.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / 'a4_cos_vs_ortho_improvement.png'}")

    # ── Save summary ────────────────────────────────────────────────
    with open(OUT_DIR / "a4_summary.txt", "w") as f:
        f.write("A4: Delta Direction Analysis — ScanNet vs TUM\n")
        f.write("=" * 60 + "\n\n")
        for dataset in ["tum", "scannet"]:
            if dataset not in all_results:
                continue
            stats = all_results[dataset]['scene_stats']
            cos_means = [s['cos_mean'] for s in stats]
            cos_stds = [s['cos_std'] for s in stats]
            drift_means = [s['drift_energy_mean'] for s in stats]
            pct_negs = [s['pct_negative'] for s in stats]
            f.write(f"--- {dataset.upper()} ({len(stats)} scenes) ---\n")
            f.write(f"  cos mean:        {np.mean(cos_means):.4f} ± {np.std(cos_means):.4f}\n")
            f.write(f"  cos std (intra): {np.mean(cos_stds):.4f} ± {np.std(cos_stds):.4f}\n")
            f.write(f"  drift energy:    {np.mean(drift_means):.4f} ± {np.std(drift_means):.4f}\n")
            f.write(f"  % negative cos:  {np.mean(pct_negs):.1f}% ± {np.std(pct_negs):.1f}%\n\n")

        f.write("\nPer-scene details:\n")
        f.write(f"{'Dataset':<10} {'Scene':<50} {'cos_mean':>8} {'cos_std':>8} {'drift_e':>8} {'neg%':>6}\n")
        f.write("-" * 96 + "\n")
        for dataset in ["tum", "scannet"]:
            if dataset not in all_results:
                continue
            for s in all_results[dataset]['scene_stats']:
                f.write(f"{dataset:<10} {s['scene']:<50} {s['cos_mean']:>8.4f} {s['cos_std']:>8.4f} "
                       f"{s['drift_energy_mean']:>8.4f} {s['pct_negative']:>5.1f}%\n")

    print(f"  Saved: {OUT_DIR / 'a4_summary.txt'}")

    # Save raw data
    np.savez(str(OUT_DIR / "a4_all_data.npz"),
             **{f"{d}_cosines": all_results[d]['all_cosines']
                for d in all_results},
             **{f"{d}_stats": np.array([(s['cos_mean'], s['cos_std'], s['drift_energy_mean'], s['pct_negative'])
                                        for s in all_results[d]['scene_stats']])
                for d in all_results})
    print(f"  Saved: {OUT_DIR / 'a4_all_data.npz'}")


if __name__ == "__main__":
    main()
