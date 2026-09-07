"""DDD3R method diagram.

Three panels:
  (a) Geometric decomposition of one per-token delta against the tracked
      drift direction.
  (b) The reweighting itself, i.e. what DDD3R actually does to the two
      components, shown as ghost-vs-solid so the asymmetry is visible.
  (c) Pipeline, grouped into the three stages of the update rule.

Design: solid blocks with white text, colourblind-safe Wong palette, no
tinted page backgrounds, sans-serif throughout.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

OUT_DIR = Path(__file__).resolve().parents[1] / "paper" / "fig"

C_DELTA = "#D55E00"   # raw delta, vermillion
C_DRIFT = "#CC79A7"   # drift-aligned component, reddish purple
C_ORTHO = "#0072B2"   # orthogonal component, deep blue
C_FINAL = "#009E73"   # regulated delta, bluish green
C_DARK  = "#1E2A38"   # decompose block
C_GATE  = "#9CA3AF"   # gate / EMA input
C_AXIS  = "#B9BEC6"   # drift axis
C_ARROW = "#6B7280"   # connectors
C_BAND  = "#F1F3F5"   # stage band
C_INK   = "#111827"
C_MUTE  = "#8A9099"

rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8.5,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "axes.linewidth": 0,
})


def varrow(ax, start, end, color, lw=2.4, ls="-", head_length=11,
           head_width=8, alpha=1.0, zorder=3):
    ax.add_patch(FancyArrowPatch(
        start, end,
        arrowstyle=f"-|>,head_length={head_length},head_width={head_width}",
        color=color, lw=lw, alpha=alpha, linestyle=ls,
        mutation_scale=1.0, zorder=zorder, capstyle="round",
        joinstyle="round"))


def carrow(ax, start, end, color=C_ARROW, lw=1.15, alpha=0.95, zorder=2):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>,head_length=6,head_width=4.2",
        color=color, lw=lw, alpha=alpha, mutation_scale=1.0,
        zorder=zorder, capstyle="round"))


def block(ax, x, y, w, h, label, fill, fontsize=10, text_color="white",
          bold=True, zorder=3):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.11",
        facecolor=fill, edgecolor="none", zorder=zorder))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fontsize, color=text_color,
            fontweight="bold" if bold else "normal", zorder=zorder + 1)


def band(ax, x, y, w, h, title):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.14",
        facecolor=C_BAND, edgecolor="none", zorder=0))
    ax.text(x + w / 2, y + h - 0.16, title, ha="center", va="top",
            fontsize=8.2, color=C_MUTE, fontweight="bold", zorder=1)


DX, DY = 2.55, 2.05          # raw delta
A_PERP, A_PAR = 0.5, 0.05    # paper defaults


def panel_geometry(ax):
    ax.set_xlim(-0.5, 3.75)
    ax.set_ylim(-0.62, 2.62)
    ax.set_aspect("equal")
    ax.axis("off")

    # shaded projection triangle makes the decomposition read at a glance
    ax.add_patch(Polygon([(0, 0), (DX, 0), (DX, DY)], closed=True,
                         facecolor="#0072B2", alpha=0.05, edgecolor="none",
                         zorder=0))

    varrow(ax, (0, 0), (3.45, 0), C_AXIS, lw=1.0, ls=(0, (4, 2.5)),
           head_length=7, head_width=5, zorder=1)
    ax.text(3.5, 0.03, r"$\mathbf{d}_t$", color=C_MUTE, fontsize=10.5,
            va="bottom", ha="left")

    varrow(ax, (0, 0), (DX, DY), C_DELTA, lw=2.6)
    ax.text(DX + 0.06, DY + 0.05, r"$\boldsymbol{\delta}_t$", color=C_DELTA,
            fontsize=12, va="bottom", ha="left", fontweight="bold")

    varrow(ax, (0, 0), (DX, 0), C_DRIFT, lw=2.4)
    ax.text(DX / 2, -0.20, r"$\boldsymbol{\delta}_t^{\parallel}$",
            color=C_DRIFT, fontsize=11.5, va="top", ha="center",
            fontweight="bold")

    varrow(ax, (DX, 0), (DX, DY), C_ORTHO, lw=2.4)
    ax.text(DX + 0.10, DY / 2, r"$\boldsymbol{\delta}_t^{\perp}$",
            color=C_ORTHO, fontsize=11.5, va="center", ha="left",
            fontweight="bold")

    sz = 0.13
    ax.plot([DX - sz, DX - sz, DX], [0, sz, sz], color=C_MUTE, lw=0.8,
            zorder=2)



def panel_reweight(ax):
    """Same construction as (a), with both components scaled by their gain."""
    ax.set_xlim(-0.5, 3.75)
    ax.set_ylim(-0.62, 2.62)
    ax.set_aspect("equal")
    ax.axis("off")

    varrow(ax, (0, 0), (3.45, 0), C_AXIS, lw=1.0, ls=(0, (4, 2.5)),
           head_length=7, head_width=5, zorder=1)
    ax.text(3.5, 0.03, r"$\mathbf{d}_t$", color=C_MUTE, fontsize=10.5,
            va="bottom", ha="left")

    # ghost of the unregulated decomposition, mirroring panel (a)
    varrow(ax, (0, 0), (DX, DY), C_DELTA, lw=2.2, alpha=0.16, zorder=1)
    varrow(ax, (0, 0), (DX, 0), C_DRIFT, lw=2.0, alpha=0.20, zorder=1)
    varrow(ax, (DX, 0), (DX, DY), C_ORTHO, lw=2.0, alpha=0.20, zorder=1)

    par, perp = A_PAR * DX, A_PERP * DY

    # regulated components, same construction, new lengths
    varrow(ax, (0, 0), (par, 0), C_DRIFT, lw=2.6, head_length=8,
           head_width=6)
    varrow(ax, (par, 0), (par, perp), C_ORTHO, lw=2.6)
    varrow(ax, (0, 0), (par, perp), C_FINAL, lw=2.8)

    ax.text(par - 0.10, perp + 0.10, r"$\tilde{\boldsymbol{\delta}}_t$",
            color=C_FINAL, fontsize=12, va="bottom", ha="left",
            fontweight="bold")
    ax.text(par + 0.30, perp * 0.62,
            r"$\alpha_\perp \boldsymbol{\delta}_t^{\perp}$",
            color=C_ORTHO, fontsize=10, ha="left", va="center",
            fontweight="bold")
    ax.text(par + 0.06, -0.16,
            r"$\alpha_\parallel \boldsymbol{\delta}_t^{\parallel}$",
            color=C_DRIFT, fontsize=10, ha="left", va="top",
            fontweight="bold")

    # faint labels on the ghosts
    ax.text(DX * 0.72, -0.16, r"$\boldsymbol{\delta}_t^{\parallel}$",
            color=C_DRIFT, alpha=0.42, fontsize=10, ha="center", va="top",
            fontweight="bold")
    ax.text(DX + 0.09, DY * 0.55, r"$\boldsymbol{\delta}_t^{\perp}$",
            color=C_ORTHO, alpha=0.42, fontsize=10, ha="left", va="center",
            fontweight="bold")


def panel_pipeline(ax):
    ax.set_xlim(0, 14.6)
    ax.set_ylim(-0.15, 4.15)
    ax.axis("off")

    band(ax, 3.30, 0.30, 3.05, 3.45, "decompose")
    band(ax, 6.60, 0.30, 3.05, 3.45, "reweight")
    band(ax, 9.90, 0.30, 3.45, 3.45, "gate")

    block(ax, 0.20, 2.42, 2.35, 0.80, r"raw $\boldsymbol{\Delta}_t$",
          C_DELTA, fontsize=10)
    block(ax, 0.20, 0.72, 2.35, 0.80, r"EMA $\mathbf{d}_{t-1}$",
          C_GATE, fontsize=10)

    block(ax, 3.60, 1.55, 2.45, 0.88, "project", C_DARK, fontsize=10.5)
    carrow(ax, (2.60, 2.82), (3.56, 2.20))
    carrow(ax, (2.60, 1.12), (3.56, 1.78))

    block(ax, 6.90, 2.42, 2.45, 0.80,
          r"$\alpha_\perp\,\boldsymbol{\delta}_t^{\perp}$", C_ORTHO,
          fontsize=11)
    block(ax, 6.90, 0.72, 2.45, 0.80,
          r"$\alpha_\parallel\,\boldsymbol{\delta}_t^{\parallel}$", C_DRIFT,
          fontsize=11)
    carrow(ax, (6.08, 2.20), (6.86, 2.82))
    carrow(ax, (6.08, 1.78), (6.86, 1.12))

    block(ax, 10.20, 1.55, 2.85, 0.88,
          r"$\beta_t\,\tilde{\boldsymbol{\delta}}_t$", C_FINAL, fontsize=11)
    carrow(ax, (9.38, 2.82), (10.16, 2.20))
    carrow(ax, (9.38, 1.12), (10.16, 1.78))

    # explicit state update, so the reader sees this is S_{t-1} -> S_t
    ax.text(13.42, 1.99, r"$\oplus$", fontsize=13, color=C_INK,
            ha="center", va="center")
    carrow(ax, (13.07, 1.99), (13.24, 1.99), lw=1.3)
    ax.text(13.42, 2.52, r"$\mathbf{S}_{t-1}$", fontsize=10, color=C_MUTE,
            ha="center", va="bottom")
    carrow(ax, (13.42, 2.46), (13.42, 2.18), lw=1.1)
    carrow(ax, (13.60, 1.99), (14.02, 1.99), lw=1.3)
    ax.text(14.10, 1.99, r"$\mathbf{S}_t$", fontsize=11.5, color=C_INK,
            va="center", ha="left", fontweight="bold")




def main():
    fig = plt.figure(figsize=(11.4, 2.78))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 2.55], wspace=0.06)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    panel_geometry(axes[0])
    panel_reweight(axes[1])
    panel_pipeline(axes[2])

    # Panel titles are placed in figure coordinates so that they line up
    # despite the equal-aspect vector panels having a shorter axes box.
    fig.canvas.draw()
    titles = ["(a) Decompose", r"(b) Reweight  ($\alpha_\perp \gg \alpha_\parallel$)",
              "(c) Pipeline"]
    for ax, t in zip(axes, titles):
        x0 = ax.get_position().x0
        fig.text(x0, 0.985, t, fontsize=10, fontweight="bold", color=C_INK,
                 ha="left", va="top")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "method_diagram.pdf")
    fig.savefig(OUT_DIR / "method_diagram.png", dpi=200)
    print("Saved", OUT_DIR / "method_diagram.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
