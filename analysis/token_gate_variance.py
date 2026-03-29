"""
Quick analysis: per-token gate variance on TUM.
Checks whether brake gate has meaningful spatial (token-level) selectivity
even though frame-averaged gate is nearly constant (~0.33).

Usage:
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python analysis/token_gate_variance.py
"""

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from glob import glob

BASE = Path("/home/szy/research/TTT3R")
MODEL_PATH = str(BASE / "model/cut3r_512_dpt_4_64.pth")
SIZE = 512

TUM_SCENES = [
    "rgbd_dataset_freiburg3_sitting_static",
    "rgbd_dataset_freiburg3_sitting_halfsphere",
    "rgbd_dataset_freiburg3_walking_xyz",
    "rgbd_dataset_freiburg3_walking_halfsphere",
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


def analyze_token_variance(model, views, device="cuda", tau=1.0):
    """Run inference with monkey-patched _momentum_gate to log per-token stats."""
    print("  Using hook approach...")

    # Reset
    gate_log = []
    cos_log = []

    original_momentum_gate = model._momentum_gate

    def hooked_momentum_gate(state_feat, new_state_feat, momentum_state, config):
        delta = new_state_feat - state_feat  # [1, N, D]
        if momentum_state.get('prev_delta') is None:
            momentum_state['prev_delta'] = delta.detach().clone()
            gate = torch.full_like(delta[..., 0:1], 0.5)
            gate_log.append({'mean': 0.5, 'std': 0.0, 'min': 0.5, 'max': 0.5})
            cos_log.append({'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0})
            return gate

        prev_delta = momentum_state['prev_delta']
        # Per-token cosine similarity
        cos_sim = F.cosine_similarity(delta, prev_delta, dim=-1)  # [1, N]

        tau_val = getattr(config, 'momentum_tau', 2.0)
        gate = torch.sigmoid(-tau_val * cos_sim).unsqueeze(-1)  # [1, N, 1]

        # Log per-token stats
        cos_vals = cos_sim.detach().cpu().squeeze()
        gate_vals = gate.detach().cpu().squeeze()

        cos_log.append({
            'mean': cos_vals.mean().item(),
            'std': cos_vals.std().item(),
            'min': cos_vals.min().item(),
            'max': cos_vals.max().item(),
            'values': cos_vals.numpy().copy(),
        })
        gate_log.append({
            'mean': gate_vals.mean().item(),
            'std': gate_vals.std().item(),
            'min': gate_vals.min().item(),
            'max': gate_vals.max().item(),
            'values': gate_vals.numpy().copy(),
        })

        momentum_state['prev_delta'] = delta.detach().clone()
        return gate

    # Monkey-patch (staticmethod)
    model._momentum_gate = staticmethod(hooked_momentum_gate)

    # Run with ttt3r_momentum
    old_update_type = model.config.model_update_type
    model.config.model_update_type = "ttt3r_momentum"
    model.config.momentum_tau = tau

    with torch.no_grad():
        model.forward_recurrent_lighter(views, device=device)

    # Restore
    model.config.model_update_type = old_update_type
    model._momentum_gate = original_momentum_gate

    return gate_log, cos_log


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print("Loading model...")
    model = load_model(args.device)

    print("\n=== Per-Token Gate Variance Analysis (TUM) ===\n")

    all_results = {}

    for scene in TUM_SCENES:
        img_dir = BASE / f"data/long_tum_s1/{scene}/rgb_1000"
        imgs = sorted(glob(str(img_dir / "*")))
        if not imgs:
            print(f"  {scene}: no images found")
            continue

        # Limit to 200 frames for speed
        imgs = imgs[:200]
        print(f"\n  {scene} ({len(imgs)} frames)")

        views = build_views(imgs, SIZE)
        gate_log, cos_log = analyze_token_variance(model, views, args.device)

        # Aggregate stats (skip frame 0)
        valid = [g for g in gate_log[1:] if g['std'] > 0]
        valid_cos = [c for c in cos_log[1:] if c['std'] > 0]

        if not valid:
            print(f"    No valid frames with variance")
            continue

        gate_mean_of_means = np.mean([g['mean'] for g in valid])
        gate_mean_of_stds = np.mean([g['std'] for g in valid])
        gate_mean_of_ranges = np.mean([g['max'] - g['min'] for g in valid])

        cos_mean_of_means = np.mean([c['mean'] for c in valid_cos])
        cos_mean_of_stds = np.mean([c['std'] for c in valid_cos])
        cos_mean_of_ranges = np.mean([c['max'] - c['min'] for c in valid_cos])

        print(f"    Gate:   mean={gate_mean_of_means:.4f}, "
              f"token_std={gate_mean_of_stds:.4f}, "
              f"token_range={gate_mean_of_ranges:.4f}")
        print(f"    Cosine: mean={cos_mean_of_means:.4f}, "
              f"token_std={cos_mean_of_stds:.4f}, "
              f"token_range={cos_mean_of_ranges:.4f}")

        # Also show temporal variation of token_std
        stds_over_time = [g['std'] for g in gate_log[1:]]
        print(f"    Gate std over time: mean={np.mean(stds_over_time):.4f}, "
              f"min={np.min(stds_over_time):.4f}, max={np.max(stds_over_time):.4f}")

        # Show some example frames with high/low variance
        if valid_cos:
            cos_stds = [(i+1, c['std']) for i, c in enumerate(cos_log[1:])]
            cos_stds.sort(key=lambda x: x[1], reverse=True)
            print(f"    Top-3 high cos_std frames: {cos_stds[:3]}")
            print(f"    Top-3 low  cos_std frames: {cos_stds[-3:]}")

        all_results[scene] = {
            'gate_mean': gate_mean_of_means,
            'gate_token_std': gate_mean_of_stds,
            'gate_token_range': gate_mean_of_ranges,
            'cos_mean': cos_mean_of_means,
            'cos_token_std': cos_mean_of_stds,
        }

    # Summary
    print("\n\n=== Summary ===")
    print(f"{'Scene':<50} {'Gate Mean':>10} {'Gate TokenStd':>14} {'Cos Mean':>10} {'Cos TokenStd':>14}")
    print("-" * 100)
    for scene, r in all_results.items():
        short = scene.replace("rgbd_dataset_freiburg3_", "")
        print(f"{short:<50} {r['gate_mean']:>10.4f} {r['gate_token_std']:>14.4f} "
              f"{r['cos_mean']:>10.4f} {r['cos_token_std']:>14.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
