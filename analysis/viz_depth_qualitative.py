"""Depth qualitative comparison figure for paper.

Creates side-by-side depth map comparisons: RGB | GT | cut3r | ttt3r | brake | ortho
with error maps below each predicted depth.

Usage:
    python analysis/viz_depth_qualitative.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from matplotlib.colors import Normalize
import matplotlib.cm as cm

# Paths
EVAL_BASE = "eval_results/video_depth"
DATA_BASE = "data/long_bonn_s1/rgbd_bonn_dataset"

METHODS = {
    "CUT3R": "cut3r",
    "TTT3R": "ttt3r",
    "Brake": "ttt3r_momentum_inv_t1",
    "Ortho": "ttt3r_ortho",
}

# Representative scenes and frames
# Pick frames where differences are visible (later in sequence = more over-update)
SCENES = [
    {
        "dataset": "bonn_s1_500",
        "seq": "balloon2",
        "gt_seq": "rgbd_bonn_balloon2",
        "frames": [100, 197, 357],  # t=197,357 have largest error gaps
        "gt_scale": 5000.0,  # Bonn GT depth in mm / 5000 = meters
        "label": "Bonn balloon2",
    },
    {
        "dataset": "bonn_s1_500",
        "seq": "crowd3",
        "gt_seq": "rgbd_bonn_crowd3",
        "frames": [100, 272, 397],
        "gt_scale": 5000.0,
        "label": "Bonn crowd3",
    },
]

OUTPUT_DIR = "analysis_results/depth_qualitative"


def load_pred_depth(dataset, method_dir, seq, frame_idx):
    """Load predicted depth (NPY) for a given method and frame."""
    path = os.path.join(EVAL_BASE, dataset, method_dir, seq, f"frame_{frame_idx:04d}.npy")
    if os.path.exists(path):
        return np.load(path)
    return None


def load_gt_depth(gt_seq, frame_idx, gt_scale):
    """Load GT depth for Bonn dataset."""
    if gt_seq is None:
        return None
    depth_dir = os.path.join(DATA_BASE, gt_seq, "depth_500")
    if not os.path.exists(depth_dir):
        return None
    files = sorted(os.listdir(depth_dir))
    if frame_idx >= len(files):
        return None
    gt = np.array(Image.open(os.path.join(depth_dir, files[frame_idx])))
    gt = gt.astype(np.float32) / gt_scale
    gt[gt == 0] = np.nan
    return gt


def load_rgb(gt_seq, frame_idx):
    """Load RGB image for Bonn dataset."""
    if gt_seq is None:
        return None
    rgb_dir = os.path.join(DATA_BASE, gt_seq, "rgb_500")
    if not os.path.exists(rgb_dir):
        return None
    files = sorted(os.listdir(rgb_dir))
    if frame_idx >= len(files):
        return None
    return np.array(Image.open(os.path.join(rgb_dir, files[frame_idx])))


def compute_error_map(pred, gt):
    """Compute absolute relative error map."""
    if gt is None or pred is None:
        return None
    # Resize pred to GT size if needed
    if pred.shape != gt.shape:
        pred_resized = np.array(Image.fromarray(pred).resize(
            (gt.shape[1], gt.shape[0]), Image.BILINEAR))
    else:
        pred_resized = pred

    # Scale pred to match GT (median scaling)
    valid = ~np.isnan(gt) & (gt > 0) & (pred_resized > 0)
    if valid.sum() < 100:
        return None
    scale = np.median(gt[valid]) / np.median(pred_resized[valid])
    pred_scaled = pred_resized * scale

    error = np.abs(pred_scaled - gt) / (gt + 1e-6)
    error[~valid] = np.nan
    return error


def make_figure_for_scene(scene_cfg):
    """Create depth comparison figure for one scene with multiple frames."""
    dataset = scene_cfg["dataset"]
    seq = scene_cfg["seq"]
    gt_seq = scene_cfg["gt_seq"]
    frames = scene_cfg["frames"]
    gt_scale = scene_cfg["gt_scale"]
    label = scene_cfg["label"]

    has_gt = gt_seq is not None

    n_frames = len(frames)
    n_methods = len(METHODS)

    if has_gt:
        # Layout: rows = frames, cols = RGB + GT + methods (depth) + methods (error)
        n_cols = 2 + n_methods  # RGB, GT, then method depths
        n_rows = n_frames * 2  # depth row + error row per frame
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3 * n_frames * 2))
    else:
        # No GT: just RGB + method depths
        n_cols = 1 + n_methods
        n_rows = n_frames
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3 * n_frames))

    if n_frames == 1:
        axes = axes[np.newaxis, :]

    for fi, frame_idx in enumerate(frames):
        # Load RGB
        rgb = load_rgb(gt_seq, frame_idx)

        # Load GT depth
        gt_depth = load_gt_depth(gt_seq, frame_idx, gt_scale) if has_gt else None

        # Load all predicted depths
        pred_depths = {}
        for method_label, method_dir in METHODS.items():
            pred_depths[method_label] = load_pred_depth(dataset, method_dir, seq, frame_idx)

        # Determine common depth range from GT or predictions
        all_depths = [d for d in pred_depths.values() if d is not None]
        if gt_depth is not None:
            valid_gt = gt_depth[~np.isnan(gt_depth)]
            vmin, vmax = np.percentile(valid_gt, [2, 98])
        elif all_depths:
            all_vals = np.concatenate([d.flatten() for d in all_depths])
            vmin, vmax = np.percentile(all_vals, [2, 98])
        else:
            vmin, vmax = 0, 5

        depth_row = fi * 2 if has_gt else fi
        error_row = fi * 2 + 1 if has_gt else None

        col = 0

        # RGB
        ax = axes[depth_row, col]
        if rgb is not None:
            ax.imshow(rgb)
        ax.set_xticks([])
        ax.set_yticks([])
        if fi == 0:
            ax.set_title("RGB", fontsize=11, fontweight='bold')
        ax.set_ylabel(f"t={frame_idx}", fontsize=10, rotation=0, labelpad=30, va='center')

        if has_gt:
            ax_err = axes[error_row, col]
            ax_err.axis('off')

        col += 1

        # GT depth
        if has_gt:
            ax = axes[depth_row, col]
            if gt_depth is not None:
                im = ax.imshow(gt_depth, cmap='Spectral_r', vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if fi == 0:
                ax.set_title("GT Depth", fontsize=11, fontweight='bold')

            ax_err = axes[error_row, col]
            ax_err.axis('off')
            col += 1

        # Method depths and error maps
        for mi, (method_label, method_dir) in enumerate(METHODS.items()):
            pred = pred_depths[method_label]

            # Depth map
            ax = axes[depth_row, col + mi]
            if pred is not None:
                ax.imshow(pred, cmap='Spectral_r', vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if fi == 0:
                ax.set_title(method_label, fontsize=11, fontweight='bold')

            # Error map
            if has_gt and gt_depth is not None:
                error = compute_error_map(pred, gt_depth)
                ax_err = axes[error_row, col + mi]
                if error is not None:
                    err_im = ax_err.imshow(error, cmap='hot', vmin=0, vmax=0.5)
                    # Add mean error text
                    mean_err = np.nanmean(error)
                    ax_err.text(0.02, 0.95, f"AbsRel={mean_err:.3f}",
                               transform=ax_err.transAxes, fontsize=8,
                               color='white', va='top',
                               bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
                ax_err.set_xticks([])
                ax_err.set_yticks([])
                if fi == 0:
                    ax_err.set_ylabel("Error", fontsize=9, rotation=0, labelpad=30, va='center') if mi == 0 else None

    fig.suptitle(f"{label}", fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    return fig


def make_compact_figure():
    """Create a paper-quality figure with best frames showing clear method differences.

    Layout: 2 rows (balloon2 t=197, t=357), each with depth + error
    Columns: RGB | GT | CUT3R | TTT3R | Brake | Ortho
    """
    configs = [
        {"dataset": "bonn_s1_500", "seq": "balloon2", "gt_seq": "rgbd_bonn_balloon2",
         "frame": 197, "gt_scale": 5000.0, "label": "t=197"},
        {"dataset": "bonn_s1_500", "seq": "balloon2", "gt_seq": "rgbd_bonn_balloon2",
         "frame": 357, "gt_scale": 5000.0, "label": "t=357"},
    ]

    n_methods = len(METHODS)
    n_cols = 2 + n_methods  # RGB + GT + 4 methods
    n_rows = len(configs) * 2  # depth + error per scene

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.0 * n_rows))

    for si, cfg in enumerate(configs):
        frame_idx = cfg["frame"]
        rgb = load_rgb(cfg["gt_seq"], frame_idx)
        gt_depth = load_gt_depth(cfg["gt_seq"], frame_idx, cfg["gt_scale"])

        pred_depths = {}
        for method_label, method_dir in METHODS.items():
            pred_depths[method_label] = load_pred_depth(
                cfg["dataset"], method_dir, cfg["seq"], frame_idx)

        # Depth range from GT
        if gt_depth is not None:
            valid_gt = gt_depth[~np.isnan(gt_depth)]
            vmin, vmax = np.percentile(valid_gt, [2, 98])
        else:
            vmin, vmax = 0, 5

        # For predictions: compute scaled depth range
        # Scale each prediction to GT, then use common range
        pred_scaled = {}
        for method_label, pred in pred_depths.items():
            if pred is not None and gt_depth is not None:
                pred_r = np.array(Image.fromarray(pred).resize(
                    (gt_depth.shape[1], gt_depth.shape[0]), Image.BILINEAR))
                valid = ~np.isnan(gt_depth) & (gt_depth > 0) & (pred_r > 0)
                if valid.sum() > 100:
                    scale = np.median(gt_depth[valid]) / np.median(pred_r[valid])
                    pred_scaled[method_label] = pred_r * scale
                else:
                    pred_scaled[method_label] = pred_r
            elif pred is not None:
                pred_scaled[method_label] = np.array(Image.fromarray(pred).resize(
                    (480, 640), Image.BILINEAR))

        depth_row = si * 2
        error_row = si * 2 + 1

        # RGB
        ax = axes[depth_row, 0]
        if rgb is not None:
            ax.imshow(rgb)
        ax.set_xticks([])
        ax.set_yticks([])
        if si == 0:
            ax.set_title("RGB", fontsize=11, fontweight='bold')
        ax.set_ylabel(cfg["label"], fontsize=11, fontweight='bold',
                      rotation=0, labelpad=35, va='center')

        ax_err = axes[error_row, 0]
        ax_err.axis('off')
        if si == 0:
            pass  # no error label needed for RGB

        # GT
        ax = axes[depth_row, 1]
        if gt_depth is not None:
            ax.imshow(gt_depth, cmap='Spectral_r', vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        if si == 0:
            ax.set_title("GT Depth", fontsize=11, fontweight='bold')
        axes[error_row, 1].axis('off')

        # Methods
        for mi, (method_label, method_dir) in enumerate(METHODS.items()):
            col = 2 + mi
            pred_s = pred_scaled.get(method_label)

            # Depth map (scaled to GT range)
            ax = axes[depth_row, col]
            if pred_s is not None:
                ax.imshow(pred_s, cmap='Spectral_r', vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if si == 0:
                ax.set_title(method_label, fontsize=11, fontweight='bold')

            # Error map
            ax_err = axes[error_row, col]
            if pred_s is not None and gt_depth is not None:
                valid = ~np.isnan(gt_depth) & (gt_depth > 0)
                error = np.full_like(gt_depth, np.nan)
                error[valid] = np.abs(pred_s[valid] - gt_depth[valid]) / (gt_depth[valid] + 1e-6)

                # Use consistent error range
                err_vmax = 0.3
                ax_err.imshow(error, cmap='hot', vmin=0, vmax=err_vmax)
                mean_err = np.nanmean(error)
                ax_err.text(0.03, 0.92, f"{mean_err:.3f}",
                           transform=ax_err.transAxes, fontsize=9,
                           color='white', va='top', fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.8))
            ax_err.set_xticks([])
            ax_err.set_yticks([])

    fig.suptitle("Bonn balloon2 — Depth Prediction Quality", fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout(pad=0.3)
    return fig


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # First check GT depth scale by comparing with predictions
    print("Checking GT depth scale...")
    gt = load_gt_depth("rgbd_bonn_balloon2", 200, 5000.0)
    pred = load_pred_depth("bonn_s1_500", "cut3r", "balloon2", 200)
    if gt is not None and pred is not None:
        valid = ~np.isnan(gt) & (gt > 0)
        print(f"  GT range: {np.nanpercentile(gt, [2, 50, 98])}")
        print(f"  Pred range: {np.percentile(pred, [2, 50, 98])}")
        # Try different scales
        for scale in [1.0, 1000.0, 5000.0]:
            gt_s = load_gt_depth("rgbd_bonn_balloon2", 200, scale)
            valid_s = ~np.isnan(gt_s) & (gt_s > 0)
            ratio = np.median(gt_s[valid_s]) / np.median(pred[pred > 0])
            print(f"  Scale {scale}: GT median={np.nanmedian(gt_s):.3f}, ratio={ratio:.3f}")

    # Compact paper figure
    print("\nGenerating compact comparison figure...")
    fig = make_compact_figure()
    fig.savefig(os.path.join(OUTPUT_DIR, "depth_comparison_compact.png"),
                dpi=200, bbox_inches='tight')
    fig.savefig(os.path.join(OUTPUT_DIR, "depth_comparison_compact.pdf"),
                bbox_inches='tight')
    print(f"  Saved to {OUTPUT_DIR}/depth_comparison_compact.png/pdf")
    plt.close(fig)

    # Per-scene detailed figures
    for scene_cfg in SCENES:
        if scene_cfg["gt_seq"] is None:
            print(f"\nSkipping {scene_cfg['label']} (no GT depth)")
            continue
        print(f"\nGenerating figure for {scene_cfg['label']}...")
        fig = make_figure_for_scene(scene_cfg)
        name = scene_cfg["seq"]
        fig.savefig(os.path.join(OUTPUT_DIR, f"depth_{name}_detail.png"),
                    dpi=200, bbox_inches='tight')
        fig.savefig(os.path.join(OUTPUT_DIR, f"depth_{name}_detail.pdf"),
                    bbox_inches='tight')
        print(f"  Saved to {OUTPUT_DIR}/depth_{name}_detail.png/pdf")
        plt.close(fig)

    print("\nDone!")


if __name__ == "__main__":
    main()
