"""Generate a 2-panel method diagram for DDD3R.

Left:  Geometric decomposition of a per-token delta into drift / orthogonal
       components, with the reweighted result and an explanatory caption.
Right: Pipeline (decompose -> reweight -> gate) with light-fill blocks and
       deliberate connector arrows.

Saves to paper/fig/method_diagram.{pdf,png}.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT_DIR = Path("/home/szy/research/TTT3R/paper/fig")

# ── Color palette ──────────────────────────────────────────────────────
# Pastel fills + saturated strokes for blocks; saturated arrows for vectors.
def light(hex_color, alpha=0.18):
    """Return an rgba tuple used as a soft fill."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255
    return (r, g, b, alpha)


C_DELTA = "#D4877F"      # raw delta (warm coral)
C_DRIFT = "#7E6CB0"      # drift component (muted purple)
C_ORTHO = "#3F86C9"      # orthogonal component (clear blue)
C_FINAL = "#4FA15A"      # reweighted delta (forest green)
C_GATE  = "#7A7A7A"      # gating step (neutral gray)
C_DIR   = "#999999"      # auxiliary drift axis (light gray)
C_INK   = "#222222"      # high-contrast text


rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.6,
})


# ─────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────
def vector_arrow(ax, start, end, color, lw=2.0, alpha=1.0, ls="-",
                 head_length=10, head_width=7, zorder=3):
    a = FancyArrowPatch(
        start, end,
        arrowstyle=f"-|>,head_length={head_length},head_width={head_width}",
        color=color, lw=lw, alpha=alpha, linestyle=ls,
        mutation_scale=1.0, zorder=zorder, capstyle="round",
    )
    ax.add_patch(a)


def connector_arrow(ax, start, end, color="#9A9A9A", lw=1.0):
    a = FancyArrowPatch(
        start, end,
        arrowstyle="-|>,head_length=6,head_width=4",
        color=color, lw=lw, alpha=0.95,
        mutation_scale=1.0, zorder=2, capstyle="round",
    )
    ax.add_patch(a)


def soft_block(ax, x, y, w, h, label, stroke, fontsize=8.5, italic=False):
    """Light-fill block with a colored stroke and centered label."""
    fill = light(stroke, alpha=0.16)
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.025,rounding_size=0.10",
        facecolor=fill, edgecolor=stroke, linewidth=1.0,
        joinstyle="round",
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2, y + h / 2, label,
        ha="center", va="center",
        fontsize=fontsize, color=stroke,
        fontweight="bold" if not italic else "normal",
        style="italic" if italic else "normal",
    )


# ─────────────────────────────────────────────────────────────────────
# Panel (a): Geometric decomposition
# ─────────────────────────────────────────────────────────────────────
def panel_geometric(ax):
    ax.set_xlim(-0.55, 4.4)
    ax.set_ylim(-0.55, 3.4)
    ax.set_aspect("equal")
    ax.axis("off")

    # Soft tinted background to delineate the panel
    ax.add_patch(Rectangle(
        (-0.55, -0.55), 4.95, 3.95,
        facecolor=light("#F2F2F2", alpha=0.45), edgecolor="none",
        zorder=0,
    ))

    # Drift direction d_t (long, dashed light gray)
    vector_arrow(ax, (0, 0), (3.9, 0), C_DIR,
                 lw=1.0, ls=(0, (4, 2.5)), head_length=7, head_width=5,
                 zorder=1)
    ax.text(3.95, 0.05, r"$\mathbf{d}_t$",
            color=C_DIR, fontsize=10.5, va="bottom", ha="left",
            fontweight="bold")

    # Raw delta delta_t (coral)
    delta_x, delta_y = 2.6, 2.2
    vector_arrow(ax, (0, 0), (delta_x, delta_y), C_DELTA, lw=2.4)
    ax.text(delta_x + 0.05, delta_y + 0.08,
            r"$\boldsymbol{\delta}_t$ (raw)",
            color=C_DELTA, fontsize=10.5, va="bottom", ha="left",
            fontweight="bold")

    # Parallel component delta_para (purple, on x-axis)
    vector_arrow(ax, (0, 0), (delta_x, 0), C_DRIFT, lw=2.2)
    ax.text(delta_x / 2, -0.27, r"$\boldsymbol{\delta}_t^{\parallel}$  (drift)",
            color=C_DRIFT, fontsize=10, va="top", ha="center",
            fontweight="bold")

    # Perpendicular component (blue, vertical)
    vector_arrow(ax, (delta_x, 0), (delta_x, delta_y), C_ORTHO, lw=2.2)
    ax.text(delta_x + 0.10, delta_y / 2,
            r"$\boldsymbol{\delta}_t^{\perp}$  (novel)",
            color=C_ORTHO, fontsize=10, va="center", ha="left",
            fontweight="bold")

    # Right-angle marker
    sz = 0.14
    ax.plot([delta_x - sz, delta_x - sz, delta_x],
            [0, sz, sz], color="#7A7A7A", lw=0.8, zorder=2)

    # Reweighted delta tilde (green)  with alpha_perp >> alpha_para
    a_perp, a_par = 0.6, 0.10
    tilde_x, tilde_y = a_par * delta_x, a_perp * delta_y
    vector_arrow(ax, (0, 0), (tilde_x, tilde_y), C_FINAL, lw=2.6)
    ax.text(tilde_x - 0.18, tilde_y / 2,
            r"$\tilde{\boldsymbol{\delta}}_t$",
            color=C_FINAL, fontsize=11, va="center", ha="right",
            fontweight="bold")
    ax.text(
        2.0, 2.95,
        r"$\tilde{\boldsymbol{\delta}}_t = \alpha_\perp \boldsymbol{\delta}_t^{\perp} + \alpha_\parallel \boldsymbol{\delta}_t^{\parallel}$",
        fontsize=9.5, color=C_FINAL, ha="center", va="top",
        fontweight="bold",
    )

    # Annotation strip at top
    ax.text(
        2.0, 3.32,
        r"$\alpha_\perp \!\gg\! \alpha_\parallel$:  attenuate drift, preserve novelty",
        fontsize=8.5, color=C_INK, ha="center", va="top", style="italic",
    )

    ax.set_title("(a) Directional decomposition",
                 fontsize=10, loc="left", fontweight="bold", pad=2,
                 color=C_INK)


# ─────────────────────────────────────────────────────────────────────
# Panel (b): Pipeline diagram
# ─────────────────────────────────────────────────────────────────────
def panel_pipeline(ax):
    ax.set_xlim(0, 13.2)
    ax.set_ylim(-0.4, 4.0)
    ax.axis("off")

    # Soft band background covering the whole panel
    ax.add_patch(Rectangle(
        (0, -0.4), 13.2, 4.4,
        facecolor=light("#F2F2F2", alpha=0.45), edgecolor="none",
        zorder=0,
    ))

    # Inputs (left column)
    soft_block(ax, 0.2, 2.55, 2.4, 0.75,
               r"raw $\boldsymbol{\Delta}_t$",
               C_DELTA, fontsize=9.5)
    soft_block(ax, 0.2, 0.55, 2.4, 0.75,
               r"EMA $\mathbf{d}_{t-1}$",
               C_DIR, fontsize=9.5)

    # Stage 1: Decompose
    soft_block(ax, 3.65, 1.55, 2.5, 0.85, "Decompose", "#3A3A3A",
               fontsize=10)

    connector_arrow(ax, (2.65, 2.92), (3.62, 2.20))
    connector_arrow(ax, (2.65, 0.92), (3.62, 1.78))

    # Outputs of decompose
    soft_block(ax, 6.95, 2.55, 1.45, 0.75, r"$\boldsymbol{\delta}_t^{\perp}$", C_ORTHO,
               fontsize=10.5)
    soft_block(ax, 6.95, 0.55, 1.45, 0.75, r"$\boldsymbol{\delta}_t^{\parallel}$", C_DRIFT,
               fontsize=10.5)
    connector_arrow(ax, (6.18, 2.20), (6.92, 2.92))
    connector_arrow(ax, (6.18, 1.78), (6.92, 0.92))

    # Stage 2: Reweight
    soft_block(ax, 9.20, 1.55, 2.7, 0.85,
               r"$\alpha_\perp \boldsymbol{\delta}_t^{\perp}\!+\!\alpha_\parallel \boldsymbol{\delta}_t^{\parallel}$",
               C_FINAL, fontsize=9.5)
    connector_arrow(ax, (8.43, 2.92), (9.17, 2.20))
    connector_arrow(ax, (8.43, 0.92), (9.17, 1.78))

    # Stage 3: Gate (× β_t) — small inline block
    soft_block(ax, 9.40, 0.10, 2.30, 0.65,
               r"$\times\, \beta_t$  token gate",
               C_GATE, fontsize=9, italic=True)
    connector_arrow(ax, (10.55, 1.50), (10.55, 0.78))

    # Output: state update arrow with label
    out_x = 12.55
    connector_arrow(ax, (11.74, 0.42), (out_x - 0.05, 0.42))
    ax.text(out_x, 0.42,
            r"$\to \mathbf{S}_t$",
            color=C_INK, fontsize=10, va="center", ha="left",
            fontweight="bold")

    ax.set_title("(b) Pipeline:  decompose $\\to$ reweight $\\to$ gate",
                 fontsize=10, loc="left", fontweight="bold", pad=2,
                 color=C_INK)


# ─────────────────────────────────────────────────────────────────────
# Compose figure
# ─────────────────────────────────────────────────────────────────────
def main():
    fig = plt.figure(figsize=(10.6, 2.65))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 2.2], wspace=0.10)
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
