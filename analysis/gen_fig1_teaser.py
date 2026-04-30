"""Generate Figure 1 (teaser) for DDD3R paper.

Layout:
  (a) 3D colored point cloud (bird's eye) with GT/CUT3R/TTT3R/DDD3R trajectories
  (b) Gate collapse visualization (β_t over time, multiple scenes)
  (c) Method overview diagram — TTT3R problem vs DDD3R solution

Point cloud from ScanNet GT depth + GT poses.
Scene0759_00: CUT3R 1.10 → TTT3R 0.34 → DDD3R_brake 0.12
"""

import matplotlib
import numpy as np

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image

NEURIPS_TEXTWIDTH = 5.5


def setup_style():
    rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.2,
            "patch.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "mathtext.fontset": "stix",
        }
    )


OUT_DIR = Path("/home/szy/research/TTT3R/paper/fig")
DATA_ROOT = Path("/home/szy/research/TTT3R")

# Colors
C_GT = "#2ECC40"
C_CUT3R_TRAJ = "#FF4136"
C_TTT3R_TRAJ = "#FF851B"
C_BRAKE_TRAJ = "#0074D9"

SCENE = "scene0759_00"
SCENE_DIR = DATA_ROOT / f"data/long_scannet_s3/{SCENE}"


def load_replica_poses(filepath):
    """Load replica format (16 floats = 4x4 per line)."""
    poses = []
    positions = []
    with open(filepath) as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            if len(vals) == 16:
                mat = np.array(vals).reshape(4, 4)
                if not np.any(np.isinf(mat)):
                    poses.append(mat)
                    positions.append(mat[:3, 3])
    return poses, np.array(positions)


def load_tum_traj(filepath):
    data = np.loadtxt(filepath)
    return data[:, 1:4]


def align_sim3(pred_pos, gt_pos):
    pm, gm = pred_pos.mean(0), gt_pos.mean(0)
    pc, gc = pred_pos - pm, gt_pos - gm
    H = pc.T @ gc
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    pr = (R @ pc.T).T
    s = np.sum(gc * pr) / np.sum(pr * pr)
    return s * (R @ pred_pos.T).T + gm - s * R @ pm


def build_scannet_pointcloud(scene_dir, every_n=5, max_points=300000):
    """Build colored point cloud from ScanNet GT depth + GT pose."""
    fx, fy, cx, cy = 577.87, 577.87, 319.5, 239.5

    pose_file = scene_dir / "pose_1000.txt"
    depth_dir = scene_dir / "depth_1000"
    color_dir = scene_dir / "color_1000"

    poses, _ = load_replica_poses(pose_file)
    depth_files = sorted(depth_dir.glob("frame_*.png"))
    color_files = sorted(color_dir.glob("frame_*.jpg"))
    n_frames = min(len(depth_files), len(color_files), len(poses))

    all_pts = []
    all_colors = []
    indices = list(range(0, n_frames, every_n))
    pts_per_frame = max(max_points // len(indices), 500)

    for idx in indices:
        if idx >= n_frames:
            continue
        depth_img = np.array(Image.open(depth_files[idx]))
        depth = depth_img.astype(np.float64) / 1000.0
        color_img = Image.open(color_files[idx]).resize((640, 480), Image.LANCZOS)
        color = np.array(color_img)

        h, w = depth.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        valid = (depth > 0.1) & (depth < 8.0)
        valid_idx = np.where(valid.ravel())[0]

        if len(valid_idx) > pts_per_frame:
            valid_idx = np.random.choice(valid_idx, pts_per_frame, replace=False)

        u_v = u.ravel()[valid_idx]
        v_v = v.ravel()[valid_idx]
        d_v = depth.ravel()[valid_idx]

        x = (u_v - cx) * d_v / fx
        y = (v_v - cy) * d_v / fy
        z = d_v
        pts_cam = np.stack([x, y, z], axis=1)

        T = poses[idx]
        pts_world = (T[:3, :3] @ pts_cam.T).T + T[:3, 3]
        colors_v = color.reshape(-1, 3)[valid_idx] / 255.0

        all_pts.append(pts_world)
        all_colors.append(colors_v)

    pts = np.concatenate(all_pts)
    colors = np.concatenate(all_colors)

    if len(pts) > max_points:
        idx = np.random.choice(len(pts), max_points, replace=False)
        pts = pts[idx]
        colors = colors[idx]

    return pts, colors


def draw_method_diagram(ax):
    """Draw method overview: TTT3R (left) vs DDD3R (right), side by side."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.2)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # ── Shared style ──
    def add_box(x, y, w, h, label, fc, ec, fs=7, bold=False):
        box = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.06",
            facecolor=fc, edgecolor=ec, linewidth=0.7, zorder=3,
        )
        ax.add_patch(box)
        weight = "bold" if bold else "normal"
        ax.text(
            x + w / 2, y + h / 2, label, fontsize=fs,
            ha="center", va="center", zorder=4, color="#333333",
            fontweight=weight,
        )

    def add_arrow(x1, y1, x2, y2):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->, head_width=0.15, head_length=0.1",
                            color="#888888", lw=0.7, shrinkA=1, shrinkB=1),
        )

    # ── Divider ──
    ax.plot([5, 5], [0.1, 3.1], color="#CCCCCC", linewidth=0.6,
            linestyle="--", zorder=0)

    # ════════════════════════════════════
    # LEFT: TTT3R — "gate collapses"
    # ════════════════════════════════════
    cx_l = 2.5  # center x

    # Title — black badge style
    ax.text(cx_l, 3.1, "TTT3R", fontsize=7, fontweight="bold",
            ha="center", va="top", color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#888888", alpha=0.8))

    # S_{t-1} → CUT3R → β_t · δ_t → S_t
    add_box(0.3, 1.6, 1.0, 0.55, "$\\mathbf{S}_{t\\!-\\!1}$", "#F5F5F5", "#BBBBBB")
    add_box(1.8, 1.6, 1.3, 0.55, "Decoder", "#F5F5F5", "#BBBBBB")
    add_box(3.6, 1.6, 1.1, 0.55, "$\\mathbf{S}_t$", "#F5F5F5", "#BBBBBB")

    add_arrow(1.3, 1.87, 1.8, 1.87)   # S → Decoder
    add_arrow(3.1, 1.87, 3.6, 1.87)   # Decoder → S_t

    # β_t box (small, above arrow)
    add_box(3.15, 2.25, 0.8, 0.38, "$\\beta_t$", "#FFE0E0", "#D4877F", fs=6.5)

    # Arrow from β_t down to the connection
    add_arrow(3.55, 2.25, 3.55, 2.05)

    # δ_t label on arrow
    ax.text(2.95, 2.0, "$\\boldsymbol{\\delta}_t$", fontsize=6,
            ha="center", va="bottom", color="#888888")

    # X_t input arrow
    ax.text(2.45, 2.55, "$\\mathbf{X}_t$", fontsize=6.5,
            ha="center", va="bottom", color="#666666")
    add_arrow(2.45, 2.5, 2.45, 2.15)

    # Problem annotation (red text below)
    ax.text(cx_l, 0.65, "$\\beta_t \\approx 0.33$  (constant)",
            fontsize=7, ha="center", va="center", color="#C0392B",
            fontstyle="italic")
    ax.text(cx_l, 0.25,
            "$\\mathbf{S}_t = \\mathbf{S}_{t-1} + \\beta_t \\boldsymbol{\\delta}_t$",
            fontsize=7, ha="center", va="center", color="#999999")

    # ════════════════════════════════════
    # RIGHT: DDD3R — directional decomposition
    # ════════════════════════════════════
    cx_r = 7.5
    ox = 5.2  # offset

    # Title — black badge style
    ax.text(cx_r, 3.1, "DDD3R (ours)", fontsize=7, fontweight="bold",
            ha="center", va="top", color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#2980B9", alpha=0.85))

    # S_{t-1} → Decoder → split → δ⊥ + δ∥ → S_t
    add_box(ox + 0.1, 1.6, 1.0, 0.55, "$\\mathbf{S}_{t\\!-\\!1}$", "#F5F5F5", "#BBBBBB")
    add_box(ox + 1.6, 1.6, 1.2, 0.55, "Decoder", "#F5F5F5", "#BBBBBB")

    # Fork point
    fork_x = ox + 3.05

    # Two branches
    add_box(fork_x + 0.3, 2.15, 1.2, 0.42,
            "$\\alpha_\\perp \\boldsymbol{\\delta}_\\perp$",
            "#E8F4FD", "#2980B9", fs=6.5)
    add_box(fork_x + 0.3, 1.15, 1.2, 0.42,
            "$\\alpha_\\parallel \\boldsymbol{\\delta}_\\parallel$",
            "#FDEDEC", "#E74C3C", fs=6.5)

    # S_t output
    add_box(fork_x + 1.85, 1.6, 1.0, 0.55, "$\\mathbf{S}_t$", "#F5F5F5", "#BBBBBB")

    # Arrows
    add_arrow(ox + 1.1, 1.87, ox + 1.6, 1.87)  # S → Decoder
    add_arrow(ox + 2.8, 1.97, fork_x + 0.3, 2.36)   # Decoder → perp
    add_arrow(ox + 2.8, 1.77, fork_x + 0.3, 1.36)   # Decoder → par
    add_arrow(fork_x + 1.5, 2.36, fork_x + 1.85, 2.0)  # perp → S_t
    add_arrow(fork_x + 1.5, 1.36, fork_x + 1.85, 1.72)  # par → S_t

    # δ_t label at fork
    ax.text(fork_x + 0.05, 1.87, "$\\boldsymbol{\\delta}_t$", fontsize=6,
            ha="center", va="center", color="#888888")

    # β_t box
    add_box(ox + 2.85, 2.30, 0.65, 0.35, "$\\beta_t$", "#E8F4FD", "#2980B9", fs=6)
    add_arrow(ox + 3.17, 2.30, ox + 3.17, 2.05)

    # X_t input
    ax.text(ox + 2.2, 2.55, "$\\mathbf{X}_t$", fontsize=6.5,
            ha="center", va="bottom", color="#666666")
    add_arrow(ox + 2.2, 2.5, ox + 2.2, 2.15)

    # α labels
    ax.text(fork_x + 1.65, 2.55, "$\\alpha_\\perp \\!>\\! \\alpha_\\parallel$",
            fontsize=6, ha="center", va="bottom", color="#2980B9",
            fontweight="bold")

    # Equation below
    ax.text(cx_r, 0.65, "decompose $\\rightarrow$ reweight $\\rightarrow$ gate",
            fontsize=6.5, ha="center", va="center", color="#2980B9",
            fontstyle="italic")
    ax.text(cx_r, 0.25,
            "$\\mathbf{S}_t = \\mathbf{S}_{t-1} + \\beta_t"
            "(\\alpha_\\perp \\boldsymbol{\\delta}_\\perp +"
            " \\alpha_\\parallel \\boldsymbol{\\delta}_\\parallel)$",
            fontsize=7, ha="center", va="center", color="#2980B9")


def draw_method_compact(ax):
    """Compact method diagram for right column: TTT3R (top) vs DDD3R (bottom)."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.6)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    lx = 0.0   # badge left edge
    cx = 5.0   # center for equations

    # ── Divider ──
    ax.plot([0.0, 10.0], [2.0, 2.0], color="#DDDDDD", linewidth=0.5,
            linestyle="--", zorder=0)

    # ════ TOP: TTT3R ════
    ax.text(lx, 3.45, "TTT3R", fontsize=6.5, fontweight="bold",
            ha="left", va="top", color="white",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#888888", alpha=0.8))
    ax.text(cx, 2.8,
            "$\\mathbf{S}_t = \\mathbf{S}_{t-1} + \\beta_t \\boldsymbol{\\delta}_t$",
            fontsize=8, ha="center", va="center", color="#777777")
    ax.text(cx, 2.2,
            "$\\beta_t \\approx 0.33$ — Dampening",
            fontsize=6, ha="center", va="center", color="#C0392B",
            fontweight="bold")

    # ════ BOTTOM: DDD3R ════
    ax.text(lx, 1.65, "DDD3R (ours)", fontsize=6.5, fontweight="bold",
            ha="left", va="top", color="white",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#2980B9", alpha=0.85))
    ax.text(cx, 1.0,
            "$\\mathbf{S}_t = \\mathbf{S}_{t-1} + \\beta_t"
            "(\\alpha_\\perp \\boldsymbol{\\delta}_\\perp +"
            " \\alpha_\\parallel \\boldsymbol{\\delta}_\\parallel)$",
            fontsize=7.5, ha="center", va="center", color="#2980B9")
    ax.text(cx, 0.4,
            "$\\alpha_\\perp, \\alpha_\\parallel$ — Directional Decomposition",
            fontsize=6, ha="center", va="center", color="#2980B9",
            fontweight="bold")


def load_sample_frames(scene_dir, n_frames=8):
    """Load evenly-spaced sample frames from the scene."""
    color_dir = scene_dir / "color_1000"
    color_files = sorted(color_dir.glob("frame_*.jpg"))
    total = len(color_files)
    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    frames = []
    for idx in indices:
        img = Image.open(color_files[idx])
        # Crop to 4:3 center and resize small
        w, h = img.size
        target_h = int(w * 3 / 4)
        if target_h < h:
            top = (h - target_h) // 2
            img = img.crop((0, top, w, top + target_h))
        img = img.resize((160, 120), Image.LANCZOS)
        frames.append((idx, np.array(img)))
    return frames


def fig_teaser():
    """Figure 1: frame strip + 3D trajectory + gate collapse + method diagram."""
    print("Building ScanNet point cloud...")
    pts, colors = build_scannet_pointcloud(SCENE_DIR, every_n=5, max_points=250000)
    print(f"  {len(pts)} points")

    # Load sample frames
    sample_frames = load_sample_frames(SCENE_DIR, n_frames=10)

    # Load GT trajectory
    gt_poses, gt_pos = load_replica_poses(SCENE_DIR / "pose_1000.txt")

    # Load predictions
    methods = [
        ("CUT3R", "cut3r", C_CUT3R_TRAJ, "--", 1.0),
        ("TTT3R", "ttt3r", C_TTT3R_TRAJ, "--", 1.2),
        ("DDD3R", "ddd3r_brake", C_BRAKE_TRAJ, "-", 1.8),
    ]

    aligned_trajs = {}
    for label, method, color, ls, lw in methods:
        traj_path = (
            DATA_ROOT
            / f"eval_results/relpose/scannet_s3_1000/{method}/{SCENE}/pred_traj.txt"
        )
        if traj_path.exists():
            pos = load_tum_traj(traj_path)
            n = min(len(pos), len(gt_pos))
            aligned = align_sim3(pos[:n], gt_pos[:n])
            aligned_trajs[label] = (aligned, color, ls, lw)

    # ── Figure layout ──
    # Row 0: frame strip (full width, thin)
    # Row 1: (a) 3D scene (left) + (b) gate collapse (right)
    # Row 2: (c) method diagram (full width)
    fig = plt.figure(figsize=(NEURIPS_TEXTWIDTH, 3.8))

    # Frame strip: show first 3, dots, last 3
    n_imgs = len(sample_frames)
    show_left = sample_frames[:5]
    show_right = sample_frames[-5:]

    strip_h = 0.10
    strip_top = 0.90
    img_w = 0.085
    img_gap = 0.006
    dots_w = 0.05  # width reserved for "..."

    # Total width: 10 images + dots gap
    total_w = 10 * (img_w + img_gap) + dots_w
    strip_left = (1.0 - total_w) / 2 + 0.04  # shift right a bit for label

    # Draw left 5 frames
    for i, (fidx, img) in enumerate(show_left):
        ax_img = fig.add_axes([
            strip_left + i * (img_w + img_gap),
            strip_top, img_w, strip_h,
        ])
        ax_img.imshow(img)
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        for spine in ax_img.spines.values():
            spine.set_edgecolor("#CCCCCC")
            spine.set_linewidth(0.4)

    # Dots in the middle
    dots_x = strip_left + 5 * (img_w + img_gap) + dots_w / 2
    fig.text(
        dots_x, strip_top + strip_h / 2,
        "$\\cdots$", fontsize=10, ha="center", va="center", color="#999999",
    )

    # Draw right 5 frames
    right_start = strip_left + 5 * (img_w + img_gap) + dots_w
    for i, (fidx, img) in enumerate(show_right):
        ax_img = fig.add_axes([
            right_start + i * (img_w + img_gap),
            strip_top, img_w, strip_h,
        ])
        ax_img.imshow(img)
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        for spine in ax_img.spines.values():
            spine.set_edgecolor("#CCCCCC")
            spine.set_linewidth(0.4)

    # (removed t label)
    # "Input Frames" label
    fig.text(
        strip_left - 0.015, strip_top + strip_h / 2,
        "Input\nFrames", fontsize=5.5, ha="right", va="center",
        color="#666666", linespacing=1.1,
    )

    # Panel positions [left, bottom, width, height]
    # (a) left full height, right column split: (b) top + (c) bottom
    row1_bot = 0.10
    row1_h = 0.78
    right_left = 0.73
    right_w = 0.26
    gate_h = 0.32
    gate_bot = row1_bot + row1_h - gate_h - 0.02  # pull down from top
    method_h = row1_h - gate_h - 0.08       # bottom of right column
    method_bot = row1_bot

    ax_scene = fig.add_axes([0.0, row1_bot, 0.68, row1_h])                # (a)
    ax_gate = fig.add_axes([right_left, gate_bot, right_w, gate_h])       # (b)
    ax_method = fig.add_axes([right_left, method_bot, right_w, method_h]) # (c)

    # ── (a) 3D scene with trajectories ──
    dark_colors = colors * 0.55 + 0.15
    ax_scene.scatter(
        pts[:, 0], pts[:, 1], c=dark_colors, s=0.06, alpha=0.35,
        rasterized=True, zorder=1,
    )
    ax_scene.plot(
        gt_pos[:, 0], gt_pos[:, 1], "-", color=C_GT,
        linewidth=2.0, label="Ground truth", zorder=5, alpha=0.9,
    )
    z = 6
    for label, (pos, color, ls, lw) in aligned_trajs.items():
        ax_scene.plot(
            pos[:, 0], pos[:, 1], ls, color=color,
            linewidth=lw, label=label, zorder=z,
        )
        z += 1
    ax_scene.scatter(
        [gt_pos[0, 0]], [gt_pos[0, 1]], marker="*", s=90,
        color="yellow", zorder=15, edgecolors="black", linewidth=0.5,
    )
    ax_scene.set_aspect("equal")
    # Tight crop around the point cloud + trajectories
    # Compute data bounds from points and GT trajectory
    all_x = np.concatenate([pts[:, 0], gt_pos[:, 0]])
    all_y = np.concatenate([pts[:, 1], gt_pos[:, 1]])
    xlo, xhi = np.percentile(all_x, 1), np.percentile(all_x, 99)
    ylo, yhi = np.percentile(all_y, 1), np.percentile(all_y, 99)
    pad = 0.15  # small padding
    ax_scene.set_xlim(xlo - pad, xhi + pad)
    ax_scene.set_ylim(ylo - pad, yhi + pad)
    ax_scene.set_xticks([])
    ax_scene.set_yticks([])
    for spine in ax_scene.spines.values():
        spine.set_visible(False)

    leg = ax_scene.legend(
        frameon=True, framealpha=0.9, edgecolor="#CCCCCC", fontsize=5.5,
        loc="upper left", handlelength=1.5, borderpad=0.4,
        labelspacing=0.25, facecolor="white",
    )
    leg.get_frame().set_linewidth(0.4)

    # ── Panel labels + title badges — all on the same horizontal line ──
    label_y_fig = row1_bot + row1_h - 0.015

    # Consistent spacing: label then gap then badge
    lab_badge_gap = 0.05  # gap between (x) and badge

    # (a) label
    a_lab_x = 0.01
    fig.text(a_lab_x, label_y_fig, "(a)", fontsize=8, fontweight="bold", va="bottom")

    # (b) label + badge
    b_lab_x = right_left - 0.03
    fig.text(b_lab_x, label_y_fig, "(b)", fontsize=8, fontweight="bold", va="bottom")
    fig.text(
        b_lab_x + lab_badge_gap, label_y_fig, "Gate collapse",
        fontsize=6.5, fontweight="bold",
        va="bottom", ha="left", color="white",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.6),
    )
    # (c) label + badge
    method_label_y = method_bot + method_h + 0.005
    c_lab_x = b_lab_x  # align with (b)
    fig.text(c_lab_x, method_label_y, "(c)", fontsize=8, fontweight="bold", va="bottom")
    fig.text(
        c_lab_x + lab_badge_gap, method_label_y, "Method overview",
        fontsize=6.5, fontweight="bold",
        va="bottom", ha="left", color="white",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.6),
    )
    # (a) ATE bar — inside, bottom center
    ax_scene.text(
        0.5, 0.02,
        "ATE:  CUT3R $\\mathbf{1.10}$  |  TTT3R $\\mathbf{0.34}$  |  "
        "DDD3R $\\mathbf{0.12}$",
        transform=ax_scene.transAxes, fontsize=6, ha="center", va="bottom",
        color="white",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.5),
    )
    # (a) panel label already placed above

    # ── (b) Gate collapse ──
    # Average gate curves per dataset (interpolated to common x-axis)
    gate_groups = {
        "ScanNet avg": (
            ["a1_scannet_scene0707_00.npz", "a1_scannet_scene0710_00.npz",
             "a1_scannet_scene0758_00.npz"],
            "#3182BD",  # blue
        ),
        "KITTI avg": (
            ["a1_kitti_2011_09_26_drive_0005_sync_02.npz",
             "a1_kitti_2011_09_26_drive_0023_sync_02.npz",
             "a1_kitti_2011_09_26_drive_0036_sync_02.npz"],
            "#E6550D",  # orange
        ),
    }
    for label, (fnames, color) in gate_groups.items():
        # Load all sequences, interpolate to 500-point common axis
        all_interp = []
        for fname in fnames:
            fpath = DATA_ROOT / f"analysis_results/a1a2_dynamics/{fname}"
            if fpath.exists():
                d = np.load(fpath)
                gates = d["gates"]
                x_orig = np.linspace(0, 1, len(gates))
                x_common = np.linspace(0, 1, 500)
                all_interp.append(np.interp(x_common, x_orig, gates))
        if all_interp:
            avg_gates = np.mean(all_interp, axis=0)
            x_pct = np.linspace(0, 100, len(avg_gates))
            ax_gate.plot(x_pct, avg_gates, color=color, linewidth=0.7,
                         alpha=0.8, label=label)
            mu = avg_gates.mean()
            ax_gate.axhline(mu, color=color, linewidth=0.5, linestyle="--",
                            alpha=0.5)

    ax_gate.axhline(1/3, color="#666666", linewidth=1.0, linestyle="--",
                    alpha=0.8, zorder=4)
    ax_gate.text(
        0.97, 1/3 - 0.003, "$\\approx\\!0.33$",
        transform=ax_gate.get_yaxis_transform(),
        fontsize=7, ha="right", va="top", color="#333333",
        fontweight="bold",
    )

    ax_gate.set_ylim(0.27, 0.42)
    ax_gate.set_xlabel("")
    ax_gate.set_ylabel("")
    ax_gate.set_title("")

    # Clean axis style: no spines, minimal ticks
    for spine in ax_gate.spines.values():
        spine.set_visible(False)
    ax_gate.set_xticks([])  # no x-axis — just "Frame →" label
    ax_gate.set_yticks([0.30, 0.33, 0.36, 0.39])
    ax_gate.tick_params(left=True, bottom=False, labelleft=True, labelbottom=False,
                        length=2, width=0.4, color="#CCCCCC", labelcolor="#AAAAAA",
                        labelsize=5.5)
    ax_gate.grid(True, alpha=0.08, linewidth=0.3)

    # Axis labels inside plot
    ax_gate.text(0.95, 0.04, "Frame $\\rightarrow$", transform=ax_gate.transAxes,
                 fontsize=6.5, color="#999999", ha="right", va="bottom")
    ax_gate.text(0.18, 0.97, "$\\uparrow$", transform=ax_gate.transAxes,
                 fontsize=7, color="#999999", ha="center", va="top")
    ax_gate.text(0.18, 0.86, "$\\beta_t$", transform=ax_gate.transAxes,
                 fontsize=7, color="#999999", ha="center", va="top")

    # Legend — upper right
    leg_g = ax_gate.legend(
        frameon=True, framealpha=0.9, edgecolor="#CCCCCC", fontsize=5,
        loc="upper right", borderpad=0.3, labelspacing=0.2,
    )
    leg_g.get_frame().set_linewidth(0.3)

    # ── (c) Compact method diagram (right-bottom) ──
    draw_method_compact(ax_method)

    fig.savefig(OUT_DIR / "fig1_teaser.pdf")
    fig.savefig(OUT_DIR / "fig1_teaser.png", dpi=300)
    print("Saved fig1_teaser.pdf/.png")
    plt.close(fig)


if __name__ == "__main__":
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_teaser()
