"""
S4: Gate Activation Visualization
==================================

Records per-frame gate values (ttt3r_mask, SIASU alpha, geo gate g_geo)
for ttt3r_joint on selected ScanNet scenes. Produces time-series plots
showing how each gate component varies over the video.

Usage
-----
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python analysis/s4_gate_visualization.py \
    --model_path model/cut3r_512_dpt_4_64.pth \
    --output_dir analysis_results/s4_gate_viz \
    --size 512 --max_frames 200
"""

import os
import sys
import argparse
import json

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from add_ckpt_path import add_path_to_dust3r
from analysis.spectral_analysis import load_img_paths, build_views


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="model/cut3r_512_dpt_4_64.pth")
    p.add_argument("--scannet_root", default="data/long_scannet_s3")
    p.add_argument("--output_dir", default="analysis_results/s4_gate_viz")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--max_frames", type=int, default=200)
    p.add_argument("--frame_interval", type=int, default=1)
    # Select scenes (comma-separated scene dirs, or 'auto' for diverse set)
    p.add_argument("--scenes", default="auto",
                    help="Comma-sep scene names or 'auto'")
    return p.parse_args()


def discover_scenes(scannet_root):
    """Find all scene dirs with color_1000/ subfolder."""
    scenes = []
    if not os.path.isdir(scannet_root):
        return scenes
    for d in sorted(os.listdir(scannet_root)):
        # Try color_1000 (preprocessed) then color (raw)
        for cname in ("color_1000", "color"):
            color_dir = os.path.join(scannet_root, d, cname)
            if os.path.isdir(color_dir):
                scenes.append((d, color_dir))
                break
    return scenes


def select_diverse_scenes(scenes, n=3):
    """Pick scenes spread across the list for diversity."""
    if len(scenes) <= n:
        return scenes
    # Pick first, middle, last
    indices = [0, len(scenes) // 2, len(scenes) - 1]
    return [scenes[i] for i in indices[:n]]


def run_scene(model, scene_name, color_dir, args, device):
    """Run ttt3r_joint on one scene, return gate log."""
    img_paths = load_img_paths(color_dir, args.frame_interval, args.max_frames)
    if len(img_paths) < 10:
        print(f"  [skip] {scene_name}: only {len(img_paths)} frames")
        return None

    views = build_views(img_paths, args.size)
    print(f"  {scene_name}: {len(views)} frames")

    # Enable gate logging
    model._gate_log = []

    with torch.no_grad():
        ress, _ = model.forward_recurrent_lighter(views, device=device)

    gate_log = model._gate_log
    model._gate_log = None

    if not gate_log:
        print(f"  [warn] {scene_name}: no gate log entries")
        return None

    # Extract per-frame statistics
    records = []
    for entry in gate_log:
        ttt3r = entry['ttt3r_mask'].squeeze(-1).squeeze(0)  # [768]
        alpha = entry['alpha'].squeeze(-1).squeeze(0)         # [768]
        g_geo = entry['g_geo']
        eff = entry['effective'].squeeze(-1).squeeze(0)       # [768]

        # g_geo might be scalar or tensor
        if isinstance(g_geo, torch.Tensor):
            g_geo_val = g_geo.item() if g_geo.numel() == 1 else g_geo.mean().item()
        else:
            g_geo_val = float(g_geo)

        records.append({
            'frame': entry['frame'],
            'ttt3r_mean': ttt3r.mean().item(),
            'ttt3r_std': ttt3r.std().item(),
            'ttt3r_min': ttt3r.min().item(),
            'ttt3r_max': ttt3r.max().item(),
            'alpha_mean': alpha.mean().item(),
            'alpha_std': alpha.std().item(),
            'alpha_min': alpha.min().item(),
            'alpha_max': alpha.max().item(),
            'g_geo': g_geo_val,
            'effective_mean': eff.mean().item(),
            'effective_std': eff.std().item(),
        })

    return records


def plot_scene(records, scene_name, output_dir):
    """Create visualization for one scene."""
    frames = [r['frame'] for r in records]

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f'Gate Activations: {scene_name}', fontsize=14)

    # 1. ttt3r_mask
    ax = axes[0]
    means = [r['ttt3r_mean'] for r in records]
    stds = [r['ttt3r_std'] for r in records]
    ax.plot(frames, means, 'b-', linewidth=1, label='mean')
    ax.fill_between(frames,
                     [m - s for m, s in zip(means, stds)],
                     [m + s for m, s in zip(means, stds)],
                     alpha=0.2, color='blue')
    ax.set_ylabel('ttt3r_mask')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. SIASU alpha
    ax = axes[1]
    means = [r['alpha_mean'] for r in records]
    stds = [r['alpha_std'] for r in records]
    ax.plot(frames, means, 'r-', linewidth=1, label='mean')
    ax.fill_between(frames,
                     [m - s for m, s in zip(means, stds)],
                     [m + s for m, s in zip(means, stds)],
                     alpha=0.2, color='red')
    ax.set_ylabel('SIASU alpha')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Geo gate
    ax = axes[2]
    g_geos = [r['g_geo'] for r in records]
    ax.plot(frames, g_geos, 'g-', linewidth=1, label='g_geo')
    ax.set_ylabel('Geo gate')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. Effective mask
    ax = axes[3]
    means = [r['effective_mean'] for r in records]
    stds = [r['effective_std'] for r in records]
    ax.plot(frames, means, 'k-', linewidth=1, label='mean')
    ax.fill_between(frames,
                     [m - s for m, s in zip(means, stds)],
                     [m + s for m, s in zip(means, stds)],
                     alpha=0.2, color='gray')
    ax.set_ylabel('Effective mask')
    ax.set_xlabel('Frame')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, f'{scene_name}_gates.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def print_summary(all_records):
    """Print aggregate statistics across all scenes."""
    print("\n" + "=" * 60)
    print("AGGREGATE GATE STATISTICS")
    print("=" * 60)

    all_ttt3r_means = []
    all_alpha_means = []
    all_alpha_stds = []
    all_g_geos = []
    all_eff_means = []

    for scene_name, records in all_records.items():
        for r in records:
            all_ttt3r_means.append(r['ttt3r_mean'])
            all_alpha_means.append(r['alpha_mean'])
            all_alpha_stds.append(r['alpha_std'])
            all_g_geos.append(r['g_geo'])
            all_eff_means.append(r['effective_mean'])

    print(f"\nttt3r_mask (per-frame mean across tokens):")
    print(f"  mean={np.mean(all_ttt3r_means):.4f}, std={np.std(all_ttt3r_means):.4f}")
    print(f"  range=[{np.min(all_ttt3r_means):.4f}, {np.max(all_ttt3r_means):.4f}]")

    print(f"\nSIASU alpha (per-frame mean across tokens):")
    print(f"  mean={np.mean(all_alpha_means):.4f}, std={np.std(all_alpha_means):.4f}")
    print(f"  range=[{np.min(all_alpha_means):.4f}, {np.max(all_alpha_means):.4f}]")
    print(f"  avg token-level std={np.mean(all_alpha_stds):.4f}")

    print(f"\nGeo gate g_geo (scalar per frame):")
    print(f"  mean={np.mean(all_g_geos):.4f}, std={np.std(all_g_geos):.4f}")
    print(f"  range=[{np.min(all_g_geos):.4f}, {np.max(all_g_geos):.4f}]")

    print(f"\nEffective mask (per-frame mean):")
    print(f"  mean={np.mean(all_eff_means):.4f}, std={np.std(all_eff_means):.4f}")
    print(f"  range=[{np.min(all_eff_means):.4f}, {np.max(all_eff_means):.4f}]")

    # Key diagnostic: is alpha approximately constant?
    print(f"\n--- KEY DIAGNOSTIC ---")
    avg_alpha_token_std = np.mean(all_alpha_stds)
    avg_alpha_frame_std = np.std(all_alpha_means)
    print(f"Alpha token-level variance (within frame): {avg_alpha_token_std:.4f}")
    print(f"Alpha frame-level variance (across frames): {avg_alpha_frame_std:.4f}")
    if avg_alpha_token_std < 0.05 and avg_alpha_frame_std < 0.05:
        print("  -> Alpha is approximately CONSTANT (low token & frame variance)")
        print("  -> This explains why ttt3r_random performs similarly!")
    elif avg_alpha_token_std < 0.05:
        print("  -> Alpha varies across frames but NOT across tokens")
        print("  -> Behaves like a per-frame scalar, not per-token adaptive")
    else:
        print("  -> Alpha has meaningful per-token variance")
        print("  -> Frequency-domain provides genuinely adaptive gating")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    add_path_to_dust3r(args.model_path)
    from dust3r.model import ARCroco3DStereo

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ARCroco3DStereo.from_pretrained(args.model_path).to(device)
    model.eval()
    model.config.model_update_type = "ttt3r_joint"
    model.config.spectral_temperature = 1.0
    model.config.geo_gate_tau = 2.0
    model.config.geo_gate_freq_cutoff = 4

    # Discover and select scenes
    all_scenes = discover_scenes(args.scannet_root)
    if not all_scenes:
        print(f"[error] No scenes found in {args.scannet_root}")
        return

    if args.scenes == "auto":
        scenes = select_diverse_scenes(all_scenes, n=3)
    else:
        names = set(args.scenes.split(","))
        scenes = [(n, c) for n, c in all_scenes if n in names]

    print(f"[S4] Running gate visualization on {len(scenes)} scenes")

    all_records = {}
    for scene_name, color_dir in scenes:
        records = run_scene(model, scene_name, color_dir, args, device)
        if records:
            all_records[scene_name] = records
            plot_scene(records, scene_name, args.output_dir)
            # Save raw data
            json_path = os.path.join(args.output_dir, f'{scene_name}_gates.json')
            with open(json_path, 'w') as f:
                json.dump(records, f, indent=2)

    if all_records:
        print_summary(all_records)

        # Save summary
        summary_path = os.path.join(args.output_dir, 'summary.txt')
        import io
        buf = io.StringIO()
        sys.stdout = buf
        print_summary(all_records)
        sys.stdout = sys.__stdout__
        with open(summary_path, 'w') as f:
            f.write(buf.getvalue())
        print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
