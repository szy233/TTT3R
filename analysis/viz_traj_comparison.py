"""
Trajectory comparison visualization for paper figures.
Style: dark background, BEV projection on highest-variance axes.
Matches TTT3R paper Figure 16 style, extended with Brake and DD3R.

Usage:
    python analysis/viz_traj_comparison.py
"""
import sys
sys.path.insert(0, "eval/relpose")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path
from copy import deepcopy
from scipy.spatial.transform import Rotation

from evo.core import sync
from evo.core.trajectory import PoseTrajectory3D

# ── Config ──────────────────────────────────────────────────────────────
DATASET = "scannet_s3_1000"
RESULT_BASE = Path(f"eval_results/relpose/{DATASET}")
GT_BASE = Path("data/long_scannet_s3")

METHODS = [
    ("CUT3R", "cut3r",                   "#9E8EC0"),  # muted purple
    ("TTT3R", "ttt3r",                    "#D4877F"),  # muted red/pink
    ("Brake", "ttt3r_momentum_inv_t1",    "#7BBF7B"),  # muted green
    ("DD3R",  "ttt3r_ortho_an05_ad005",   "#6BAED6"),  # muted blue
]

# Scenes with largest brake improvement + high CUT3R ATE (clear drift)
SEQUENCES = [
    "scene0806_00",   # cut3r 0.86 → brake 0.10 (-89%), clean loop
    "scene0721_00",   # cut3r 2.35 → brake 0.28 (-88%), clear drift
    "scene0781_00",   # cut3r 1.53 → brake 0.15 (-90%)
    "scene0760_00",   # cut3r 0.82 → brake 0.12 (-85%)
]

OUTPUT_DIR = Path("analysis_results/traj_comparison")
AXIS_LABELS = ["X (m)", "Y (m)", "Z (m)"]

# ── Helpers ─────────────────────────────────────────────────────────────
def load_pred_traj(config_dir: str, seq: str):
    path = RESULT_BASE / config_dir / seq / "pred_traj.txt"
    if not path.exists():
        return None
    data = np.loadtxt(path)
    return PoseTrajectory3D(
        positions_xyz=data[:, 1:4],
        orientations_quat_wxyz=data[:, 4:8],
        timestamps=data[:, 0],
    )

def load_gt_traj_scannet(seq: str):
    """Load ScanNet GT from pose_1000.txt (4x4 matrices, one per line)."""
    gt_path = GT_BASE / seq / "pose_1000.txt"
    if not gt_path.exists():
        return None
    lines = gt_path.read_text().strip().split('\n')
    positions = []
    quats = []
    timestamps = []
    for i, line in enumerate(lines):
        vals = list(map(float, line.split()))
        if len(vals) != 16:
            continue
        mat = np.array(vals).reshape(4, 4)
        # Skip invalid poses (containing inf/nan)
        if not np.isfinite(mat).all():
            continue
        pos = mat[:3, 3]
        rot = Rotation.from_matrix(mat[:3, :3])
        quat_xyzw = rot.as_quat()  # scipy returns [x,y,z,w]
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        positions.append(pos)
        quats.append(quat_wxyz)
        timestamps.append(float(i))
    if len(positions) == 0:
        return None
    return PoseTrajectory3D(
        positions_xyz=np.array(positions),
        orientations_quat_wxyz=np.array(quats),
        timestamps=np.array(timestamps),
    )

def align_to_gt(pred, gt):
    pred = deepcopy(pred)
    gt = deepcopy(gt)
    if pred.timestamps.shape[0] == gt.timestamps.shape[0]:
        pred.timestamps = gt.timestamps.copy()
    gt, pred = sync.associate_trajectories(gt, pred)
    pred.align(gt, correct_scale=True)
    return pred, gt

def best_2d_axes(positions):
    var = np.var(positions, axis=0)
    axes = np.argsort(var)[::-1][:2]
    return sorted(axes)

# ── Main ────────────────────────────────────────────────────────────────
def main():
    plt.style.use('dark_background')
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.size'] = 9

    n = len(SEQUENCES)
    nrows, ncols = 2, 2
    fig, axes_grid = plt.subplots(nrows, ncols, figsize=(10, 9))
    axes_arr = axes_grid.flatten()

    for idx, seq in enumerate(SEQUENCES):
        ax = axes_arr[idx]
        gt_traj = load_gt_traj_scannet(seq)
        if gt_traj is None:
            print(f"  [WARN] GT missing for {seq}")
            ax.set_visible(False)
            continue

        gt_pos = gt_traj.positions_xyz
        ai, aj = best_2d_axes(gt_pos)

        # GT: thick semi-transparent
        ax.plot(gt_pos[:, ai], gt_pos[:, aj], '-', color='#C0C0C0',
                linewidth=3.0, label='GT', alpha=0.6, zorder=1)

        for label, config_dir, color in METHODS:
            pred = load_pred_traj(config_dir, seq)
            if pred is None:
                continue
            aligned, _ = align_to_gt(pred, gt_traj)
            pos = aligned.positions_xyz
            ax.plot(pos[:, ai], pos[:, aj], '-', color=color,
                    linewidth=1.5, label=label, alpha=0.9, zorder=2)

        ax.set_title(seq.replace("_00", ""), fontsize=11, color='white', pad=6)
        ax.set_xlabel(AXIS_LABELS[ai], fontsize=9)
        ax.set_ylabel(AXIS_LABELS[aj], fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_aspect('auto')

        if idx == 0:
            ax.legend(fontsize=8, loc='best',
                      facecolor='#2a2a2a', edgecolor='#555555',
                      framealpha=0.9)

    fig.suptitle("Estimated Camera Trajectories — ScanNet Long Sequence (1000f)",
                 fontsize=13, color='white', y=1.02)
    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        out = OUTPUT_DIR / f"scannet_traj_grid.{ext}"
        fig.savefig(out, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"Saved: {out}")
    plt.close(fig)

if __name__ == "__main__":
    main()
