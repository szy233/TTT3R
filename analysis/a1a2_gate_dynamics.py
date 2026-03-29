"""
A1: Gate Temporal Dynamics — per-frame cosine + gate + camera motion visualization
A2: Cosine Variance ↔ Improvement correlation — Var(cos) vs ATE improvement scatter

Usage:
  # A1 only (2-3 scenes, fast):
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python analysis/a1a2_gate_dynamics.py --mode a1

  # A2 (all scenes, ~2-4h on ScanNet):
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python analysis/a1a2_gate_dynamics.py --mode a2

  # Both:
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python analysis/a1a2_gate_dynamics.py --mode both
"""

import argparse
import os
import sys
import re
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob

# ── Setup ──────────────────────────────────────────────────────────────
BASE = Path("/home/szy/research/TTT3R")
OUT_DIR = BASE / "analysis_results/a1a2_dynamics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = str(BASE / "model/cut3r_512_dpt_4_64.pth")
SIZE = 512

# A1 representative scenes
A1_SCENES_SCANNET = [
    "scene0707_00",  # pick a few diverse ones
    "scene0710_00",
    "scene0758_00",
]
A1_SCENES_TUM = [
    "rgbd_dataset_freiburg3_sitting_static",
    "rgbd_dataset_freiburg3_walking_xyz",
]
A1_SCENES_KITTI = [
    "2011_09_26_drive_0023_sync_02",  # long sequence, 464 frames
    "2011_09_26_drive_0036_sync_02",  # 500 frames
    "2011_09_26_drive_0005_sync_02",  # 144 frames
]
A1_SCENES_SINTEL = [
    "alley_2",
    "ambush_4",
    "market_5",
    "temple_2",
]


def load_model(device="cuda"):
    from dust3r.model import ARCroco3DStereo
    model = ARCroco3DStereo.from_pretrained(MODEL_PATH)
    model = model.to(device)
    model.eval()
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


def load_gt_motion_scannet(scene_name, num_frames):
    """Load GT poses and compute per-frame camera motion magnitude."""
    pose_file = BASE / f"data/long_scannet_s3/{scene_name}/pose_1000.txt"
    if not pose_file.exists():
        return None
    poses = np.loadtxt(str(pose_file))  # [N, 12] or [N, 16]
    n = min(len(poses), num_frames)
    poses = poses[:n]

    motions = [0.0]  # first frame has no motion
    for t in range(1, n):
        if poses.shape[1] >= 12:
            t1 = poses[t - 1, :3] if poses.shape[1] == 3 else poses[t - 1, [3, 7, 11]]
            t2 = poses[t, :3] if poses.shape[1] == 3 else poses[t, [3, 7, 11]]
            # Extract translation from 4x4 row-major: columns 3,7,11
            if poses.shape[1] >= 12:
                t1 = np.array([poses[t-1, 3], poses[t-1, 7], poses[t-1, 11]])
                t2 = np.array([poses[t, 3], poses[t, 7], poses[t, 11]])
            trans_diff = np.linalg.norm(t2 - t1)
            motions.append(trans_diff)
    return np.array(motions[:num_frames])


def load_gt_motion_tum(scene_name, num_frames):
    """Load GT poses (TUM format) and compute per-frame camera motion."""
    pose_file = BASE / f"data/long_tum_s1/{scene_name}/groundtruth_1000.txt"
    if not pose_file.exists():
        return None
    data = np.loadtxt(str(pose_file))  # [N, 8]: timestamp, x, y, z, qx, qy, qz, qw
    n = min(len(data), num_frames)
    data = data[:n]

    motions = [0.0]
    for t in range(1, n):
        t1 = data[t - 1, 1:4]  # x, y, z
        t2 = data[t, 1:4]
        trans_diff = np.linalg.norm(t2 - t1)
        motions.append(trans_diff)
    return np.array(motions[:num_frames])


def load_gt_motion_sintel(scene_name, num_frames):
    """Load GT poses (Sintel .cam files) and compute per-frame camera motion."""
    import struct
    cam_dir = BASE / f"data/sintel/training/camdata_left/{scene_name}"
    if not cam_dir.exists():
        return None
    cam_files = sorted(glob(str(cam_dir / "*.cam")))
    n = min(len(cam_files), num_frames)
    if n < 2:
        return None

    positions = []
    for f in cam_files[:n]:
        with open(f, 'rb') as fh:
            tag = struct.unpack('f', fh.read(4))[0]
            M = np.array(struct.unpack('9d', fh.read(72))).reshape(3, 3)  # intrinsic
            N = np.array(struct.unpack('12d', fh.read(96))).reshape(3, 4)  # extrinsic
            # Camera position = -R^T @ t
            R = N[:3, :3]
            t = N[:3, 3]
            pos = -R.T @ t
            positions.append(pos)

    motions = [0.0]
    for t in range(1, n):
        motions.append(np.linalg.norm(positions[t] - positions[t-1]))
    return np.array(motions[:num_frames])


def get_scene_images(dataset, scene_name):
    """Get sorted image paths for a scene."""
    if dataset == "scannet":
        img_dir = BASE / f"data/long_scannet_s3/{scene_name}/color_1000"
    elif dataset == "tum":
        img_dir = BASE / f"data/long_tum_s1/{scene_name}/rgb_1000"
    elif dataset == "kitti":
        img_dir = BASE / f"data/long_kitti_s1/depth_selection/val_selection_cropped/image_gathered_500/{scene_name}"
    elif dataset == "sintel":
        img_dir = BASE / f"data/sintel/training/final/{scene_name}"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if not img_dir.exists():
        print(f"  WARNING: {img_dir} not found")
        return []
    imgs = sorted(glob(str(img_dir / "*")))
    return imgs


def run_analysis(model, img_paths, device="cuda"):
    """Run model in analysis mode, return cosine/gate/delta_norm histories."""
    views = build_views(img_paths, SIZE)
    with torch.no_grad():
        ress, analysis_data = model.forward_recurrent_analysis(views, device=device)
    return analysis_data


def parse_per_scene_ate(config, dataset_dir):
    """Parse _error_log.txt to get {scene: ATE}."""
    log_path = dataset_dir / config / "_error_log.txt"
    results = {}
    if not log_path.exists():
        return results
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
                results[m.group(1).strip()] = float(m.group(2))
    return results


# ══════════════════════════════════════════════════════════════════════
# A1: Gate Temporal Dynamics
# ══════════════════════════════════════════════════════════════════════
def run_a1(model, device):
    print("\n=== A1: Gate Temporal Dynamics ===")

    for dataset, scenes, load_motion_fn in [
        ("scannet", A1_SCENES_SCANNET, load_gt_motion_scannet),
        ("tum", A1_SCENES_TUM, load_gt_motion_tum),
        ("kitti", A1_SCENES_KITTI, lambda s, n: None),
        ("sintel", A1_SCENES_SINTEL, load_gt_motion_sintel),
    ]:
        for scene in scenes:
            print(f"\n  Processing {dataset}/{scene}...")
            img_paths = get_scene_images(dataset, scene)
            if not img_paths:
                continue

            # Run analysis
            data = run_analysis(model, img_paths, device)
            cosines = np.array(data['cosine_history'])
            gates = np.array(data['gate_history'])
            delta_norms = np.array(data['delta_norm_history'])
            T = len(cosines)

            # Load GT motion
            motions = load_motion_fn(scene, T)

            # Plot
            fig, axes = plt.subplots(3 if motions is not None else 2, 1,
                                      figsize=(14, 8 if motions is not None else 6),
                                      sharex=True)

            frames = np.arange(T)

            # Panel 1: Cosine similarity
            ax = axes[0]
            ax.plot(frames, cosines, color="#4477AA", linewidth=0.8, alpha=0.8)
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
            ax.set_ylabel("cos(δ_t, δ_{t-1})")
            ax.set_title(f"{dataset}/{scene} — Stability Brake Dynamics")
            ax.grid(True, alpha=0.2)

            # Panel 2: Gate value
            ax = axes[1]
            ax.plot(frames, gates, color="#228833", linewidth=0.8, alpha=0.8)
            ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)
            ax.set_ylabel("Gate α_t = σ(-τ·cos)")
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.2)

            # Panel 3: GT camera motion (if available)
            if motions is not None:
                ax = axes[2] if len(axes) > 2 else axes[1]
                motion_plot = motions[:T]
                ax.plot(frames[:len(motion_plot)], motion_plot,
                        color="#EE7733", linewidth=0.8, alpha=0.8)
                ax.set_ylabel("Camera motion (m)")
                ax.set_xlabel("Frame")
                ax.grid(True, alpha=0.2)
            else:
                axes[-1].set_xlabel("Frame")

            fig.tight_layout()
            out_path = OUT_DIR / f"a1_{dataset}_{scene}.png"
            fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"    Saved: {out_path}")

            # Also save raw data as npz
            np.savez(str(OUT_DIR / f"a1_{dataset}_{scene}.npz"),
                     cosines=cosines, gates=gates, delta_norms=delta_norms,
                     motions=motions if motions is not None else np.array([]))


# ══════════════════════════════════════════════════════════════════════
# A2: Cosine Variance ↔ Improvement Correlation
# ══════════════════════════════════════════════════════════════════════
def run_a2(model, device):
    print("\n=== A2: Cosine Variance ↔ Improvement ===")

    scannet_dir = BASE / "eval_results/relpose/scannet_s3_1000"
    tum_dir = BASE / "eval_results/relpose/tum_s1_1000"

    for dataset, eval_dir, scene_list_dir, load_motion_fn in [
        ("scannet", scannet_dir,
         BASE / "data/long_scannet_s3", load_gt_motion_scannet),
        ("tum", tum_dir,
         BASE / "data/long_tum_s1", load_gt_motion_tum),
    ]:
        print(f"\n  Dataset: {dataset}")

        # Get per-scene ATE for random and inv_t1
        random_ates = parse_per_scene_ate("ttt3r_random", eval_dir)
        brake_ates = parse_per_scene_ate("ttt3r_momentum_inv_t1", eval_dir)

        # Find common scenes
        common_scenes = sorted(set(random_ates.keys()) & set(brake_ates.keys()))
        print(f"  Common scenes with ATE data: {len(common_scenes)}")

        # Collect cosine variance for each scene
        cos_variances = []
        improvements = []
        valid_scenes = []

        for scene in common_scenes:
            img_paths = get_scene_images(dataset, scene)
            if not img_paths:
                print(f"    SKIP {scene}: no images")
                continue

            print(f"    Running {scene} ({len(img_paths)} frames)...")
            model.config.model_update_type = "ttt3r"  # run vanilla to get raw deltas
            data = run_analysis(model, img_paths, device)
            cosines = np.array(data['cosine_history'][1:])  # skip frame 0

            if len(cosines) < 2:
                print(f"    SKIP {scene}: too few frames")
                continue

            cos_var = np.var(cosines)
            cos_variances.append(cos_var)

            r_ate = random_ates[scene]
            b_ate = brake_ates[scene]
            improvement = (r_ate - b_ate) / r_ate * 100  # positive = brake better
            improvements.append(improvement)
            valid_scenes.append(scene)

            print(f"      Var(cos)={cos_var:.6f}, improvement={improvement:+.1f}%")

        cos_variances = np.array(cos_variances)
        improvements = np.array(improvements)

        if len(cos_variances) < 3:
            print(f"  Not enough data for {dataset}")
            continue

        # Compute correlation
        from scipy.stats import pearsonr, spearmanr
        r_pearson, p_pearson = pearsonr(cos_variances, improvements)
        r_spearman, p_spearman = spearmanr(cos_variances, improvements)

        print(f"\n  Pearson:  r={r_pearson:.3f}, p={p_pearson:.4f}")
        print(f"  Spearman: r={r_spearman:.3f}, p={p_spearman:.4f}")

        # Scatter plot
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        ax.scatter(cos_variances, improvements, s=40, alpha=0.7,
                   edgecolors="k", linewidths=0.5, color="#4477AA", zorder=3)

        # Fit line
        z = np.polyfit(cos_variances, improvements, 1)
        x_fit = np.linspace(cos_variances.min(), cos_variances.max(), 100)
        ax.plot(x_fit, np.polyval(z, x_fit), "r--", alpha=0.6, linewidth=1.5)

        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
        ax.set_xlabel("Var(cos(δ_t, δ_{t-1}))", fontsize=12)
        ax.set_ylabel("ATE Improvement vs constant 0.5 (%)", fontsize=12)
        ax.set_title(f"{dataset.upper()}: Cosine Variance ↔ Improvement\n"
                     f"Pearson r={r_pearson:.3f} (p={p_pearson:.3f}), "
                     f"Spearman r={r_spearman:.3f} (p={p_spearman:.3f})",
                     fontsize=11)
        ax.grid(True, alpha=0.2)

        fig.tight_layout()
        out_path = OUT_DIR / f"a2_{dataset}_cos_var_vs_improvement.png"
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")

        # Save data
        np.savez(str(OUT_DIR / f"a2_{dataset}_data.npz"),
                 cos_variances=cos_variances,
                 improvements=improvements,
                 scenes=valid_scenes,
                 pearson_r=r_pearson, pearson_p=p_pearson,
                 spearman_r=r_spearman, spearman_p=p_spearman)

        # Write to summary
        with open(OUT_DIR / f"a2_{dataset}_summary.txt", "w") as f:
            f.write(f"A2: {dataset.upper()} Cosine Variance vs Improvement\n")
            f.write(f"{'='*55}\n\n")
            f.write(f"Valid scenes: {len(valid_scenes)}\n")
            f.write(f"Pearson:  r={r_pearson:.4f}, p={p_pearson:.4f}\n")
            f.write(f"Spearman: r={r_spearman:.4f}, p={p_spearman:.4f}\n\n")
            f.write(f"{'Scene':<50s}  {'Var(cos)':>10s}  {'Improv%':>8s}\n")
            f.write("-" * 72 + "\n")
            for i, scene in enumerate(valid_scenes):
                f.write(f"{scene:<50s}  {cos_variances[i]:10.6f}  "
                        f"{improvements[i]:+8.1f}%\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="both",
                        choices=["a1", "a2", "both"])
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print("Loading model...")
    model = load_model(args.device)
    # Set momentum_tau for gate computation in analysis
    model.config.momentum_tau = 1.0

    if args.mode in ("a1", "both"):
        model.config.model_update_type = "ttt3r"
        run_a1(model, args.device)

    if args.mode in ("a2", "both"):
        run_a2(model, args.device)

    print("\nDone!")


if __name__ == "__main__":
    main()
