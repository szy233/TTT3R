"""
A4b: Drift energy at different sequence lengths for scaling curve analysis.
Reuses A4 infrastructure but parametrizes frame count.

Usage:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python analysis/a4_drift_scaling.py --dataset scannet --max_frames 200
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python analysis/a4_drift_scaling.py --dataset scannet --max_frames 500
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
from pathlib import Path
from glob import glob

BASE = Path("/home/szy/research/TTT3R")
OUT_DIR = BASE / "analysis_results/a4_drift_scaling"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = str(BASE / "model/cut3r_512_dpt_4_64.pth")
SIZE = 512


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


def get_scene_images(dataset, scene_name, max_frames):
    if dataset == "scannet":
        img_dir = BASE / f"data/long_scannet_s3/{scene_name}/color_{max_frames}"
        if not img_dir.exists():
            return []
        return sorted(glob(str(img_dir / "*")))[:max_frames]
    elif dataset == "tum":
        # TUM only has 1000f data; truncate to max_frames
        img_dir = BASE / f"data/long_tum_s1/{scene_name}/rgb_1000"
        if not img_dir.exists():
            return []
        return sorted(glob(str(img_dir / "*")))[:max_frames]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def get_scenes(dataset):
    if dataset == "scannet":
        base = BASE / "data/long_scannet_s3"
        return sorted([d.name for d in base.iterdir() if d.is_dir() and d.name.startswith("scene")])
    elif dataset == "tum":
        base = BASE / "data/long_tum_s1"
        return sorted([d.name for d in base.iterdir() if d.is_dir() and d.name.startswith("rgbd")])
    return []


def analyze_scene(model, img_paths, device="cuda"):
    views = build_views(img_paths, SIZE)
    with torch.no_grad():
        ress, analysis_data = model.forward_recurrent_analysis(views, device=device)

    cosines = np.array(analysis_data['cosine_history'])
    cos = cosines[1:]  # skip frame 0
    drift_energy = cos ** 2

    return {
        'cos_mean': float(np.mean(cos)),
        'cos_std': float(np.std(cos)),
        'drift_energy_mean': float(np.mean(drift_energy)),
        'drift_energy_std': float(np.std(drift_energy)),
        'pct_negative': float(np.mean(cos < 0) * 100),
        'n_frames': len(img_paths),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["scannet", "tum"])
    parser.add_argument("--max_frames", type=int, required=True)
    args = parser.parse_args()

    device = "cuda"
    model = load_model(device)
    scenes = get_scenes(args.dataset)
    print(f"Dataset: {args.dataset}, frames: {args.max_frames}, scenes: {len(scenes)}")

    results = []
    for si, scene in enumerate(scenes):
        img_paths = get_scene_images(args.dataset, scene, args.max_frames)
        if not img_paths or len(img_paths) < 10:
            print(f"  [{si+1}/{len(scenes)}] SKIP {scene}: {len(img_paths)} frames")
            continue

        print(f"  [{si+1}/{len(scenes)}] {scene} ({len(img_paths)}f)...", end=" ", flush=True)
        stats = analyze_scene(model, img_paths, device)
        stats['scene'] = scene
        results.append(stats)
        print(f"cos={stats['cos_mean']:.3f}, drift_e={stats['drift_energy_mean']:.3f}, neg={stats['pct_negative']:.1f}%")

    # Summary
    cos_means = [r['cos_mean'] for r in results]
    drift_means = [r['drift_energy_mean'] for r in results]
    print(f"\n=== {args.dataset} {args.max_frames}f Summary ({len(results)} scenes) ===")
    print(f"  cos mean:     {np.mean(cos_means):.4f} +/- {np.std(cos_means):.4f}")
    print(f"  drift energy: {np.mean(drift_means):.4f} +/- {np.std(drift_means):.4f}")

    # Save
    out_file = OUT_DIR / f"{args.dataset}_{args.max_frames}f.json"
    with open(out_file, "w") as f:
        json.dump({
            'dataset': args.dataset,
            'max_frames': args.max_frames,
            'n_scenes': len(results),
            'cos_mean': float(np.mean(cos_means)),
            'cos_std': float(np.std(cos_means)),
            'drift_energy_mean': float(np.mean(drift_means)),
            'drift_energy_std': float(np.std(drift_means)),
            'per_scene': results,
        }, f, indent=2)
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
