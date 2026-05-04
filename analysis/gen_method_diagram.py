"""DDD3R method diagram — minimalist redesign.

Two panels:
  (a) Geometric decomposition of one per-token delta.
  (b) Pipeline (decompose -> reweight -> gate).

Design principles: solid blocks with white text, no tinted backgrounds,
no descriptor labels, deliberate arrows, sans-serif.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_DIR = Path("/home/szy/research/TTT3R/paper/fig")

# Wong-inspired colorblind-safe palette, slightly desaturated for print.
C_DELTA = "#D55E00"   # raw delta -- vermillion
C_DRIFT = "#CC79A7"   # drift component -- reddish purple
C_ORTHO = "#0072B2"   # orthogonal component -- deep blue
C_FINAL = "#009E73"   # reweighted delta -- bluish green
C_DARK  = "#1E2A38"   # decompose block -- near-black navy
C_GATE  = "#9CA3AF"   # gate / EMA input -- slate gray
C_AXIS  = "#C9CDD2"   # drift axis dashed -- light gray
C_ARROW = "#6B7280"   # connector arrows -- mid gray
C_INK   = "#111827"

rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8.5,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "axes.linewidth": 0,
})


def varrow(ax, start, end, color, lw=2.4, ls="-",
           head_length=11, head_width=8, alpha=1.0, zorder=3):
    a = FancyArrowPatch(
        start, end,
        arrowstyle=f"-|>,head_length={head_length},head_width={head_width}",
        color=color, lw=lw, alpha=alpha, linestyle=ls,
        mutation_scale=1.0, zorder=zorder, capstyle="round",
        joinstyle="round",
    )
    ax.add_patch(a)


def carrow(ax, start, end, color=C_ARROW, lw=1.1):
    a = FancyArrowPatch(
        start, end,
        arrowstyle="-|>,head_length=6,head_width=4.2",
        color=color, lw=lw, alpha=0.9,
        mutation_scale=1.0, zorder=2, capstyle="round",
    )
    ax.add_patch(a)


def block(ax, x, y, w, h, label, fill, fontsize=10, text_color="white",
          bold=True, italic=False):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.0,rounding_size=0.10",
        facecolor=fill, edgecolor="none",
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2, y + h / 2, label,
        ha="center", va="center",
        fontsize=fontsize, color=text_color,
        fontweight="bold" if bold else "normal",
        style="italic" if italic else "normal",
    )


# ─────────────────────────────────────────────────────────────────────
# Panel (a): vectors only
# ─────────────────────────────────────────────────────────────────────
def panel_geometric(ax):
    ax.set_xlim(-0.55, 4.4)
    ax.set_ylim(-0.55, 3.4)
    ax.set_aspect("equal")
    ax.axis("off")

    # Drift axis (light dashed)
    varrow(ax, (0, 0), (3.95, 0), C_AXIS,
           lw=1.0, ls=(0, (4, 2.5)), head_length=7, head_width=5, zorder=1)
    ax.text(4.0, 0.05, r"$\mathbf{d}_t$",
            color="#888888", fontsize=10.5, va="bottom", ha="left")

    # Raw delta_t
    delta_x, delta_y = 2.6, 2.2
    varrow(ax, (0, 0), (delta_x, delta_y), C_DELTA, lw=2.6)
    ax.text(delta_x + 0.05, delta_y + 0.06,
            r"$\boldsymbol{\delta}_t$",
            color=C_DELTA, fontsize=12, va="bottom", ha="left",
            fontweight="bold")

    # Drift component (parallel)
    varrow(ax, (0, 0), (delta_x, 0), C_DRIFT, lw=2.4)
    ax.text(delta_x / 2, -0.22,
            r"$\boldsymbol{\delta}_t^{\parallel}$",
            color=C_DRIFT, fontsize=11.5, va="top", ha="center",
            fontweight="bold")

    # Orthogonal component
    varrow(ax, (delta_x, 0), (delta_x, delta_y), C_ORTHO, lw=2.4)
    ax.text(delta_x + 0.10, delta_y / 2,
            r"$\boldsymbol{\delta}_t^{\perp}$",
            color=C_ORTHO, fontsize=11.5, va="center", ha="left",
            fontweight="bold")

    # Right-angle marker
    sz = 0.13
    ax.plot([delta_x - sz, delta_x - sz, delta_x],
            [0, sz, sz], color="#999999", lw=0.7, zorder=2)

    # Reweighted delta tilde (green) — alpha_perp >> alpha_par
    a_perp, a_par = 0.6, 0.10
    tilde_x, tilde_y = a_par * delta_x, a_perp * delta_y
    varrow(ax, (0, 0), (tilde_x, tilde_y), C_FINAL, lw=2.8)
    ax.text(tilde_x - 0.18, tilde_y + 0.08,
            r"$\tilde{\boldsymbol{\delta}}_t$",
            color=C_FINAL, fontsize=12, va="bottom", ha="right",
            fontweight="bold")

    # Equation as a clean caption inside the panel
    ax.text(2.0, 3.1,
            r"$\tilde{\boldsymbol{\delta}}_t = \alpha_\perp \boldsymbol{\delta}_t^{\perp} + \alpha_\parallel \boldsymbol{\delta}_t^{\parallel},\;\; \alpha_\perp \!\gg\! \alpha_\parallel$",
            fontsize=9.5, color=C_INK, ha="center", va="top")

    ax.set_title("(a) Directional decomposition",
                 fontsize=10, loc="left", fontweight="bold", pad=2,
                 color=C_INK)


# ─────────────────────────────────────────────────────────────────────
# Panel (b): pipeline
# ─────────────────────────────────────────────────────────────────────
def panel_pipeline(ax):
    ax.set_xlim(0, 13.4)
    ax.set_ylim(0, 4.0)
    ax.axis("off")

    # Inputs
    block(ax, 0.30, 2.55, 2.30, 0.78,
          r"raw $\boldsymbol{\Delta}_t$",
          C_DELTA, fontsize=10)
    block(ax, 0.30, 0.55, 2.30, 0.78,
          r"EMA $\mathbf{d}_{t-1}$",
          C_GATE, fontsize=10)

    # Decompose (dark block)
    block(ax, 3.65, 1.55, 2.40, 0.85, "Decompose", C_DARK, fontsize=10.5)
    carrow(ax, (2.65, 2.94), (3.62, 2.20))
    carrow(ax, (2.65, 0.94), (3.62, 1.78))

    # Decomposed components
    block(ax, 6.85, 2.55, 1.40, 0.78,
          r"$\boldsymbol{\delta}_t^{\perp}$", C_ORTHO, fontsize=11)
    block(ax, 6.85, 0.55, 1.40, 0.78,
          r"$\boldsymbol{\delta}_t^{\parallel}$", C_DRIFT, fontsize=11)
    carrow(ax, (6.08, 2.20), (6.82, 2.94))
    carrow(ax, (6.08, 1.78), (6.82, 0.94))

    # Reweight + gate combined
    block(ax, 9.05, 1.55, 2.95, 0.85,
          r"$\beta_t \,(\alpha_\perp \boldsymbol{\delta}_t^{\perp} + \alpha_\parallel \boldsymbol{\delta}_t^{\parallel})$",
          C_FINAL, fontsize=10)
    carrow(ax, (8.28, 2.94), (9.02, 2.20))
    carrow(ax, (8.28, 0.94), (9.02, 1.78))

    # Output
    carrow(ax, (12.04, 1.97), (12.95, 1.97), lw=1.3)
    ax.text(13.0, 1.97, r"$\mathbf{S}_t$",
            color=C_INK, fontsize=11, va="center", ha="left",
            fontweight="bold")

    ax.set_title(
        r"(b) Pipeline: decompose $\to$ reweight $\to$ gate",
        fontsize=10, loc="left", fontweight="bold", pad=2, color=C_INK,
    )


def main():
    fig = plt.figure(figsize=(10.6, 2.55))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 2.25], wspace=0.10)
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
