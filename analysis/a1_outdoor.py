"""
A1 outdoor: Gate Temporal Dynamics on KITTI + Sintel
Run after a1a2_gate_dynamics.py A1 (indoor) is done.

Usage:
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python analysis/a1_outdoor.py
"""

import numpy as np
import torch
import struct
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob

BASE = Path("/home/szy/research/TTT3R")
OUT_DIR = BASE / "analysis_results/a1a2_dynamics"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = str(BASE / "model/cut3r_512_dpt_4_64.pth")
SIZE = 512

KITTI_SCENES = [
    "2011_09_26_drive_0023_sync_02",  # 464 frames
    "2011_09_26_drive_0036_sync_02",  # 500 frames
    "2011_09_26_drive_0005_sync_02",  # 144 frames
]
SINTEL_SCENES = [
    "alley_2",      # 50 frames
    "ambush_4",     # 33 frames
    "market_5",     # 50 frames
    "temple_2",     # 50 frames
]


def load_model(device="cuda"):
    from dust3r.model import ARCroco3DStereo
    model = ARCroco3DStereo.from_pretrained(MODEL_PATH)
    model = model.to(device).eval()
    model.config.momentum_tau = 1.0
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
            "idx": i, "instance": str(i),
            "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(0),
            "img_mask": torch.tensor(True).unsqueeze(0),
            "ray_mask": torch.tensor(False).unsqueeze(0),
            "update": torch.tensor(True).unsqueeze(0),
            "reset": torch.tensor(False).unsqueeze(0),
        }
        views.append(view)
    return views


def load_sintel_motion(scene_name, num_frames):
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
            M = np.array(struct.unpack('9d', fh.read(72))).reshape(3, 3)
            N = np.array(struct.unpack('12d', fh.read(96))).reshape(3, 4)
            R, t = N[:3, :3], N[:3, 3]
            positions.append(-R.T @ t)
    motions = [0.0]
    for t in range(1, n):
        motions.append(np.linalg.norm(positions[t] - positions[t-1]))
    return np.array(motions[:num_frames])


def run_scene(model, dataset, scene, motion_fn, device="cuda"):
    if dataset == "kitti":
        img_dir = BASE / f"data/long_kitti_s1/depth_selection/val_selection_cropped/image_gathered_500/{scene}"
    elif dataset == "sintel":
        img_dir = BASE / f"data/sintel/training/final/{scene}"
    else:
        return

    imgs = sorted(glob(str(img_dir / "*")))
    if not imgs:
        print(f"  WARNING: no images in {img_dir}")
        return

    print(f"\n  Processing {dataset}/{scene} ({len(imgs)} frames)...")
    views = build_views(imgs, SIZE)
    with torch.no_grad():
        ress, data = model.forward_recurrent_analysis(views, device=device)

    cosines = np.array(data['cosine_history'])
    gates = np.array(data['gate_history'])
    delta_norms = np.array(data['delta_norm_history'])
    T = len(cosines)

    motions = motion_fn(scene, T) if motion_fn else None

    # Plot
    n_panels = 3 if motions is not None else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 3 * n_panels), sharex=True)
    frames = np.arange(T)

    ax = axes[0]
    ax.plot(frames, cosines, color="#4477AA", linewidth=0.8, alpha=0.8)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    ax.set_ylabel("cos(δ_t, δ_{t-1})")
    ax.set_title(f"{dataset}/{scene} — Stability Brake Dynamics")
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    ax.plot(frames, gates, color="#228833", linewidth=0.8, alpha=0.8)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)
    ax.set_ylabel("Gate α_t = σ(-τ·cos)")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.2)

    if motions is not None and len(axes) > 2:
        ax = axes[2]
        ax.plot(frames[:len(motions)], motions[:T], color="#EE7733", linewidth=0.8, alpha=0.8)
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

    np.savez(str(OUT_DIR / f"a1_{dataset}_{scene}.npz"),
             cosines=cosines, gates=gates, delta_norms=delta_norms,
             motions=motions if motions is not None else np.array([]))

    # Print stats
    print(f"    cos: mean={cosines[1:].mean():.3f}, std={cosines[1:].std():.3f}, "
          f"range=[{cosines[1:].min():.3f}, {cosines[1:].max():.3f}]")
    print(f"    gate: mean={gates[1:].mean():.3f}, std={gates[1:].std():.3f}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print("Loading model...")
    model = load_model(args.device)

    print("\n=== A1 Outdoor: KITTI ===")
    for scene in KITTI_SCENES:
        run_scene(model, "kitti", scene, None, args.device)

    print("\n=== A1 Outdoor: Sintel ===")
    for scene in SINTEL_SCENES:
        run_scene(model, "sintel", scene, load_sintel_motion, args.device)

    print("\nDone!")


if __name__ == "__main__":
    main()
