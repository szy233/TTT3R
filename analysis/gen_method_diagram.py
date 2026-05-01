"""Generate a 2-panel method diagram for DDD3R.

Left:  Geometric decomposition of an update vector into drift / orthogonal
       components, with reweighting illustrated by arrow scaling.
Right: Stage flow (Decompose -> Reweight -> Gate) showing how the raw
       cross-attention delta becomes the final state update.

Saves to paper/fig/method_diagram.{pdf,png}.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_DIR = Path("/home/szy/research/TTT3R/paper/fig")

rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 9,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
})

C_DELTA = "#D4877F"   # raw delta (coral)
C_DRIFT = "#8B7BB5"   # drift component (purple)
C_ORTHO = "#3182BD"   # orthogonal component (blue)
C_FINAL = "#5BAA5B"   # reweighted delta (green)
C_DIR   = "#666666"   # drift direction (gray)


def arrow(ax, start, end, color, lw=1.6, alpha=1.0, ls="-", head=8):
    a = FancyArrowPatch(
        start, end, arrowstyle=f"-|>,head_length={head},head_width={head*0.7}",
        color=color, lw=lw, alpha=alpha, linestyle=ls,
        mutation_scale=1.0, zorder=3,
    )
    ax.add_patch(a)


def panel_geometric(ax):
    ax.set_xlim(-0.4, 4.0)
    ax.set_ylim(-0.4, 3.2)
    ax.set_aspect("equal")
    ax.axis("off")

    # Drift direction d_t (long, dashed gray)
    arrow(ax, (0, 0), (3.6, 0), C_DIR, lw=1.0, ls=(0, (4, 2)), head=6)
    ax.text(3.7, 0.05, r"$\mathbf{d}_t$", color=C_DIR, fontsize=10, va="bottom", ha="left")

    # Raw delta delta_t (coral)
    delta_x, delta_y = 2.6, 2.2
    arrow(ax, (0, 0), (delta_x, delta_y), C_DELTA, lw=2.0)
    ax.text(delta_x + 0.05, delta_y + 0.05,
            r"$\boldsymbol{\delta}_t$", color=C_DELTA, fontsize=11, va="bottom", ha="left")

    # Parallel component delta_para (purple, on x-axis)
    arrow(ax, (0, 0), (delta_x, 0), C_DRIFT, lw=2.0)
    ax.text(delta_x / 2, -0.18, r"$\boldsymbol{\delta}_t^{\parallel}$",
            color=C_DRIFT, fontsize=11, va="top", ha="center")

    # Perp component (blue, vertical)
    arrow(ax, (delta_x, 0), (delta_x, delta_y), C_ORTHO, lw=2.0)
    ax.text(delta_x + 0.08, delta_y / 2,
            r"$\boldsymbol{\delta}_t^{\perp}$", color=C_ORTHO, fontsize=11, va="center", ha="left")

    # Right-angle marker
    sz = 0.13
    ax.plot([delta_x - sz, delta_x - sz, delta_x],
            [0, sz, sz], color=C_DIR, lw=0.7)

    # Reweighted delta tilde (green dashed) with alpha_perp > alpha_para
    a_perp, a_par = 0.6, 0.18
    tilde_x, tilde_y = a_par * delta_x, a_perp * delta_y
    arrow(ax, (0, 0), (tilde_x, tilde_y), C_FINAL, lw=2.4)
    ax.text(tilde_x + 0.05, tilde_y + 0.05,
            r"$\tilde{\boldsymbol{\delta}}_t \!=\! \alpha_\perp \boldsymbol{\delta}_t^{\perp} + \alpha_\parallel \boldsymbol{\delta}_t^{\parallel}$",
            color=C_FINAL, fontsize=9, va="bottom", ha="left")

    # Hint annotation: alpha_perp > alpha_para
    ax.text(2.0, 2.95, r"$\alpha_\perp \gg \alpha_\parallel$ suppresses drift, preserves novelty",
            fontsize=8.5, color="#333333", ha="center", style="italic")

    ax.set_title(r"(a) Directional decomposition", fontsize=10, pad=4, loc="left", fontweight="bold")


def block(ax, x, y, w, h, label, color, fontsize=8.5):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor=color, edgecolor=color, alpha=0.9, lw=0,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            color="white", fontsize=fontsize, fontweight="bold")


def panel_pipeline(ax):
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.2, 3.6)
    ax.axis("off")

    # Inputs
    block(ax, 0.0, 2.4, 2.4, 0.7, r"raw $\boldsymbol{\Delta}_t$", C_DELTA)
    block(ax, 0.0, 0.5, 2.4, 0.7, r"EMA $\mathbf{d}_{t-1}$", C_DIR)

    # Stage 1: Decompose
    block(ax, 3.4, 1.45, 2.4, 0.7, "Decompose", "#444444")
    arrow(ax, (2.45, 2.75), (3.4, 1.95), "#666", lw=1.0)
    arrow(ax, (2.45, 0.85), (3.4, 1.65), "#666", lw=1.0)

    # Outputs of decompose
    block(ax, 6.5, 2.4, 1.5, 0.7, r"$\boldsymbol{\delta}_t^{\perp}$", C_ORTHO)
    block(ax, 6.5, 0.5, 1.5, 0.7, r"$\boldsymbol{\delta}_t^{\parallel}$", C_DRIFT)
    arrow(ax, (5.85, 1.95), (6.5, 2.75), "#666", lw=1.0)
    arrow(ax, (5.85, 1.65), (6.5, 0.85), "#666", lw=1.0)

    # Stage 2: Reweight
    block(ax, 9.0, 1.45, 2.6, 0.7,
          r"$\alpha_\perp \boldsymbol{\delta}_t^{\perp} \!+\! \alpha_\parallel \boldsymbol{\delta}_t^{\parallel}$", C_FINAL,
          fontsize=8)
    arrow(ax, (8.05, 2.75), (9.0, 1.95), "#666", lw=1.0)
    arrow(ax, (8.05, 0.85), (9.0, 1.65), "#666", lw=1.0)

    # Stage 3: Gate (β_t)
    ax.text(10.3, 0.85, r"$\times \beta_t$ (token gate)",
            ha="center", va="center", fontsize=8.5, style="italic", color="#333333")
    arrow(ax, (10.3, 1.45), (10.3, 1.05), "#333", lw=1.2)

    # Stage labels
    ax.text(4.6, 3.25, "Stage 1", ha="center", va="bottom",
            fontsize=8, color="#666", fontweight="bold")
    ax.text(10.3, 3.25, "Stage 2 + 3", ha="center", va="bottom",
            fontsize=8, color="#666", fontweight="bold")

    ax.set_title(r"(b) DDD3R pipeline: decompose $\rightarrow$ reweight $\rightarrow$ gate",
                 fontsize=10, pad=4, loc="left", fontweight="bold")


def main():
    fig = plt.figure(figsize=(10.0, 2.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 2.2], wspace=0.12)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])

    panel_geometric(ax_left)
    panel_pipeline(ax_right)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "method_diagram.pdf")
    fig.savefig(OUT_DIR / "method_diagram.png", dpi=200)
    print("Saved method_diagram.pdf/.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
