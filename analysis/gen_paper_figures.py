"""Generate all 4 paper figures from existing analysis data.

Outputs PDF to paper/fig/ for direct LaTeX inclusion.
NeurIPS two-column style: white background, 8pt labels, proper sizing.

Usage:
    python analysis/gen_paper_figures.py [--only gate|scatter|traj|depth]
"""
import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path
from scipy import stats

# ── Global NeurIPS style ────────────────────────────────────────────────
NEURIPS_TEXTWIDTH = 5.5  # inches (two-column)
NEURIPS_COLWIDTH = 5.5   # full width for most figures

def setup_style():
    """NeurIPS-compatible matplotlib style."""
    rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 8,
        'axes.labelsize': 9,
        'axes.titlesize': 9,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'axes.linewidth': 0.6,
        'lines.linewidth': 1.0,
        'patch.linewidth': 0.5,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

BASE = Path("/home/szy/research/TTT3R")
OUT_DIR = BASE / "paper" / "fig"

# DDD3R color palette — consistent across all figures
C_GT     = '#333333'
C_CUT3R  = '#8B7BB5'   # muted purple
C_TTT3R  = '#D4877F'   # muted red/coral
C_BRAKE  = '#5BAA5B'   # muted green
C_DDD3R  = '#3182BD'   # strong blue
C_CONST  = '#E6AB02'   # muted gold


# ═══════════════════════════════════════════════════════════════════════
# Figure 1: Gate Collapse (analysis.tex Fig 3)
# ═══════════════════════════════════════════════════════════════════════

def fig_gate_collapse():
    """TTT3R gate values over time for TUM + ScanNet — shows near-constant behavior."""
    data_dir = BASE / "analysis_results" / "a1a2_dynamics"

    scenes = [
        ("a1_tum_rgbd_dataset_freiburg3_walking_xyz.npz", "TUM walking_xyz"),
        ("a1_scannet_scene0710_00.npz", "ScanNet scene0710"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(NEURIPS_TEXTWIDTH, 1.8))

    colors = [C_TTT3R, C_DDD3R]

    for ax, (fname, title), color in zip(axes, scenes, colors):
        data = np.load(data_dir / fname)
        gates = data['gates']
        frames = np.arange(len(gates))

        mu = gates.mean()
        sigma = gates.std()

        # Plot gate trace
        ax.plot(frames, gates, color=color, alpha=0.7, linewidth=0.5, rasterized=True)

        # Mean line
        ax.axhline(mu, color='#555555', linestyle='--', linewidth=0.8, alpha=0.8)

        # ±σ band
        ax.fill_between(frames, mu - sigma, mu + sigma, color=color, alpha=0.08)

        ax.set_title(f"{title} ($\\mu$={mu:.3f}, $\\sigma$={sigma:.3f})", fontsize=8)
        ax.set_xlabel("Frame", fontsize=8)
        ax.set_ylim(0.15, 0.55)
        ax.set_xlim(0, len(gates))

    axes[0].set_ylabel("Gate value $\\beta_t$", fontsize=8)

    fig.tight_layout(w_pad=1.5)
    fig.savefig(OUT_DIR / "gate_collapse.pdf")
    fig.savefig(OUT_DIR / "gate_collapse.png", dpi=200)
    print("Saved gate_collapse.pdf")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Figure 2: Scatter — Drift Energy vs Improvement (analysis.tex Fig 4)
# ═══════════════════════════════════════════════════════════════════════

def fig_scatter_drift():
    """Per-scene scatter: drift energy vs ATE improvement for DDD3R + constant dampening."""
    import re

    EVAL_BASE = BASE / "eval_results" / "relpose" / "scannet_s3_90_first"
    A4_SUMMARY = BASE / "analysis_results" / "a4_delta_direction" / "a4_summary.txt"

    def parse_ate(filepath):
        with open(filepath) as f:
            txt = f.read()
        m = re.search(r"APE w\.r\.t\. translation.*?mean\s+([\d.]+)", txt, re.DOTALL)
        return float(m.group(1)) if m else None

    def collect_ates(config_dir):
        results = {}
        d = EVAL_BASE / config_dir
        if not d.exists():
            return results
        for f in d.iterdir():
            if f.name.endswith("_eval_metric.txt"):
                scene = f.name.replace("_eval_metric.txt", "")
                ate = parse_ate(f)
                if ate is not None:
                    results[scene] = ate
        return results

    # Parse drift energy
    drift_data = {}
    with open(A4_SUMMARY) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 5 and parts[0] == "scannet":
                drift_data[parts[1]] = float(parts[4])

    # Collect ATE for methods
    ate_cut3r = collect_ates("cut3r")
    ate_ortho = collect_ates("ttt3r_ortho")
    ate_const = collect_ates("ttt3r_random")  # constant dampening α=0.5
    ate_brake = collect_ates("ttt3r_momentum_inv_t1")

    # Common scenes
    common = set(drift_data) & set(ate_cut3r) & set(ate_ortho)
    common = sorted(common)

    de = np.array([drift_data[s] for s in common])
    base = np.array([ate_cut3r[s] for s in common])
    valid = base > 0.001
    de, base = de[valid], base[valid]
    common = [s for s, v in zip(common, valid) if v]

    # Compute improvements
    imp_ortho = np.array([(ate_ortho[s] - ate_cut3r[s]) / ate_cut3r[s] * 100 for s in common])
    imp_const = np.array([(ate_const.get(s, np.nan) - ate_cut3r[s]) / ate_cut3r[s] * 100
                          if s in ate_const else np.nan for s in common])

    # Single-panel figure for paper
    fig, ax = plt.subplots(1, 1, figsize=(NEURIPS_COLWIDTH * 0.48, 2.5))

    # Constant dampening (background, gray)
    valid_c = ~np.isnan(imp_const)
    if valid_c.any():
        ax.scatter(de[valid_c], imp_const[valid_c], c='#AAAAAA', alpha=0.35, s=18,
                   edgecolors='none', marker='s', label='Constant', zorder=1)
        s_c, i_c, r_c, p_c, _ = stats.linregress(de[valid_c], imp_const[valid_c])
        x_fit = np.linspace(de.min(), de.max(), 100)
        ax.plot(x_fit, s_c * x_fit + i_c, '--', color='#AAAAAA', linewidth=0.8, alpha=0.6)

    # DDD3R (foreground, colored by direction)
    colors_ortho = np.where(imp_ortho < 0, C_DDD3R, '#E74C3C')
    ax.scatter(de, imp_ortho, c=colors_ortho, alpha=0.55, s=22,
               edgecolors='white', linewidth=0.3, zorder=2)

    # Trend line for DDD3R
    slope, intercept, r_val, p_val, _ = stats.linregress(de, imp_ortho)
    x_fit = np.linspace(de.min(), de.max(), 100)
    ax.plot(x_fit, slope * x_fit + intercept, '-', color=C_DDD3R, linewidth=1.0, alpha=0.8,
            label=f'DDD3R ($r$={r_val:.2f}, $p$={p_val:.3f})')

    ax.axhline(0, color='#CCCCCC', linestyle='-', linewidth=0.5, zorder=0)
    ax.set_xlabel("Drift energy $\\bar{e}$")
    ax.set_ylabel("$\\Delta$ATE vs CUT3R (%)")
    ax.legend(fontsize=6, loc='upper left', framealpha=0.8, edgecolor='none')

    fig.tight_layout()
    fig.savefig(OUT_DIR / "scatter_drift_improvement.pdf")
    fig.savefig(OUT_DIR / "scatter_drift_improvement.png", dpi=200)
    print("Saved scatter_drift_improvement.pdf")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Figure 3: Trajectory Comparison (experiments.tex Fig 5)
# ═══════════════════════════════════════════════════════════════════════

def fig_traj_comparison():
    """BEV trajectory comparison on ScanNet 1000f — 2 scenes side by side."""
    sys.path.insert(0, str(BASE / "eval" / "relpose"))

    from copy import deepcopy
    from scipy.spatial.transform import Rotation
    from evo.core import sync
    from evo.core.trajectory import PoseTrajectory3D

    DATASET = "scannet_s3_1000"
    RESULT_BASE = BASE / "eval_results" / "relpose" / DATASET
    GT_BASE = BASE / "data" / "long_scannet_s3"

    METHODS = [
        ("CUT3R", "cut3r",                   C_CUT3R, 1.2, 0.6),
        ("TTT3R", "ttt3r",                    C_TTT3R, 1.0, 0.7),
        ("Brake", "ttt3r_momentum_inv_t1",    C_BRAKE, 1.0, 0.8),
        ("DDD3R", "ttt3r_ortho_an05_ad005",   C_DDD3R, 1.2, 0.9),
    ]

    # Pick 2 best scenes: clear loop with visible drift difference
    SEQUENCES = ["scene0806_00", "scene0781_00"]

    def load_pred(config_dir, seq):
        path = RESULT_BASE / config_dir / seq / "pred_traj.txt"
        if not path.exists():
            return None
        data = np.loadtxt(path)
        return PoseTrajectory3D(
            positions_xyz=data[:, 1:4],
            orientations_quat_wxyz=data[:, 4:8],
            timestamps=data[:, 0],
        )

    def load_gt(seq):
        gt_path = GT_BASE / seq / "pose_1000.txt"
        if not gt_path.exists():
            return None
        lines = gt_path.read_text().strip().split('\n')
        positions, quats, timestamps = [], [], []
        for i, line in enumerate(lines):
            vals = list(map(float, line.split()))
            if len(vals) != 16:
                continue
            mat = np.array(vals).reshape(4, 4)
            if not np.isfinite(mat).all():
                continue
            pos = mat[:3, 3]
            rot = Rotation.from_matrix(mat[:3, :3])
            q_xyzw = rot.as_quat()
            quats.append([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
            positions.append(pos)
            timestamps.append(float(i))
        return PoseTrajectory3D(
            positions_xyz=np.array(positions),
            orientations_quat_wxyz=np.array(quats),
            timestamps=np.array(timestamps),
        )

    def align(pred, gt):
        pred, gt = deepcopy(pred), deepcopy(gt)
        if pred.timestamps.shape[0] == gt.timestamps.shape[0]:
            pred.timestamps = gt.timestamps.copy()
        gt, pred = sync.associate_trajectories(gt, pred)
        pred.align(gt, correct_scale=True)
        return pred, gt

    fig, axes = plt.subplots(1, 2, figsize=(NEURIPS_TEXTWIDTH, 2.5))

    for ax, seq in zip(axes, SEQUENCES):
        gt_traj = load_gt(seq)
        if gt_traj is None:
            continue

        gt_pos = gt_traj.positions_xyz
        # Best 2D projection
        var = np.var(gt_pos, axis=0)
        ai, aj = sorted(np.argsort(var)[::-1][:2])

        # GT trajectory
        ax.plot(gt_pos[:, ai], gt_pos[:, aj], '-', color=C_GT,
                linewidth=2.5, label='GT', alpha=0.4, zorder=1)

        for label, config_dir, color, lw, alpha in METHODS:
            pred = load_pred(config_dir, seq)
            if pred is None:
                continue
            aligned, _ = align(pred, gt_traj)
            pos = aligned.positions_xyz
            ax.plot(pos[:, ai], pos[:, aj], '-', color=color,
                    linewidth=lw, label=label, alpha=alpha, zorder=2)

        ax.set_title(seq.replace("_00", ""), fontsize=9)
        ax.set_xlabel(["X (m)", "Y (m)", "Z (m)"][ai])
        ax.set_ylabel(["X (m)", "Y (m)", "Z (m)"][aj])
        ax.set_aspect('equal', adjustable='datalim')
        ax.tick_params(labelsize=6)

    # Single legend at bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(OUT_DIR / "traj_comparison.pdf")
    fig.savefig(OUT_DIR / "traj_comparison.png", dpi=200)
    print("Saved traj_comparison.pdf")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Figure 4: Depth Qualitative (experiments.tex Fig 6)
# ═══════════════════════════════════════════════════════════════════════

def fig_depth_qualitative():
    """Depth prediction comparison — wrap existing rendered compact image."""
    from PIL import Image
    import shutil

    src = BASE / "analysis_results" / "depth_qualitative" / "depth_comparison_compact.pdf"
    if not src.exists():
        print("WARN: Run viz_depth_qualitative.py first to generate depth images")
        return

    # Copy the PDF directly — it's already publication quality
    shutil.copy2(src, OUT_DIR / "depth_qualitative.pdf")
    # Also copy PNG
    src_png = src.with_suffix('.png')
    if src_png.exists():
        shutil.copy2(src_png, OUT_DIR / "depth_qualitative.png")
    print("Saved depth_qualitative.pdf (copied from analysis_results)")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

FIGURES = {
    'gate': fig_gate_collapse,
    'scatter': fig_scatter_drift,
    'traj': fig_traj_comparison,
    'depth': fig_depth_qualitative,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', type=str, default=None,
                        help='Generate only this figure: gate|scatter|traj|depth')
    args = parser.parse_args()

    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.only:
        if args.only in FIGURES:
            FIGURES[args.only]()
        else:
            print(f"Unknown figure: {args.only}. Choose from: {list(FIGURES.keys())}")
    else:
        for name, fn in FIGURES.items():
            print(f"\n{'='*40} {name} {'='*40}")
            try:
                fn()
            except Exception as e:
                print(f"ERROR generating {name}: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    main()
