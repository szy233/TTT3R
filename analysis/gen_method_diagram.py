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


def varrow(ax, start, end, color, lw=1.5, ls="-", head_length=7,
           head_width=4.5, alpha=1.0, zorder=3):
    ax.add_patch(FancyArrowPatch(
        start, end,
        arrowstyle=f"-|>,head_length={head_length},head_width={head_width}",
        color=color, lw=lw, alpha=alpha, linestyle=ls,
        mutation_scale=1.0, zorder=zorder, capstyle="round",
        joinstyle="round"))


def carrow(ax, start, end, color=C_ARROW, lw=1.0, alpha=0.95, zorder=2):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>,head_length=5,head_width=3.4",
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
    ax.set_xlim(-0.45, 3.80)
    ax.set_ylim(-0.62, 2.62)
    ax.set_aspect("equal")
    ax.axis("off")

    # shaded projection triangle makes the decomposition read at a glance
    ax.add_patch(Polygon([(0, 0), (DX, 0), (DX, DY)], closed=True,
                         facecolor="#0072B2", alpha=0.05, edgecolor="none",
                         zorder=0))

    varrow(ax, (0, 0), (3.45, 0), C_AXIS, lw=0.85, ls=(0, (4, 2.5)),
           head_length=5, head_width=3.4, zorder=1)
    ax.text(3.52, 0.03, r"$\mathbf{d}_t$", color=C_MUTE, fontsize=10.5,
            va="bottom", ha="left")

    varrow(ax, (0, 0), (DX, DY), C_DELTA, lw=1.7)
    ax.text(DX + 0.06, DY + 0.05, r"$\boldsymbol{\delta}_t$", color=C_DELTA,
            fontsize=12, va="bottom", ha="left", fontweight="bold")

    varrow(ax, (0, 0), (DX, 0), C_DRIFT, lw=1.5)
    ax.text(DX / 2, -0.20, r"$\boldsymbol{\delta}_t^{\parallel}$",
            color=C_DRIFT, fontsize=11.5, va="top", ha="center",
            fontweight="bold")

    varrow(ax, (DX, 0), (DX, DY), C_ORTHO, lw=1.5)
    ax.text(DX + 0.10, DY / 2, r"$\boldsymbol{\delta}_t^{\perp}$",
            color=C_ORTHO, fontsize=11.5, va="center", ha="left",
            fontweight="bold")

    sz = 0.13
    ax.plot([DX - sz, DX - sz, DX], [0, sz, sz], color=C_MUTE, lw=0.75,
            zorder=2)


def panel_reweight(ax):
    """Ghost of (a), then the reweighted components, then their sum.

    Drawing the sum on top of its own components would put three arrows in a
    0.13-wide band, because alpha_parallel is a twentieth of DX. Stating it
    head-to-tail and restating the resultant beside an equals sign keeps the
    gains at true scale and still legible.
    """
    ax.set_xlim(-0.45, 6.10)
    ax.set_ylim(-0.62, 2.62)
    ax.set_aspect("equal")
    ax.axis("off")

    varrow(ax, (0, 0), (5.70, 0), C_AXIS, lw=0.85, ls=(0, (4, 2.5)),
           head_length=5, head_width=3.4, zorder=1)
    ax.text(5.77, 0.03, r"$\mathbf{d}_t$", color=C_MUTE, fontsize=10.5,
            va="bottom", ha="left")

    # ghost of the unregulated decomposition, mirroring panel (a)
    varrow(ax, (0, 0), (DX, DY), C_DELTA, lw=1.5, alpha=0.18, zorder=1)
    varrow(ax, (0, 0), (DX, 0), C_DRIFT, lw=1.4, alpha=0.22, zorder=1)
    varrow(ax, (DX, 0), (DX, DY), C_ORTHO, lw=1.4, alpha=0.22, zorder=1)
    ax.text(DX * 0.5, -0.19, r"$\boldsymbol{\delta}_t^{\parallel}$",
            color=C_DRIFT, alpha=0.45, fontsize=10, ha="center", va="top",
            fontweight="bold")
    ax.text(DX + 0.09, DY * 0.5, r"$\boldsymbol{\delta}_t^{\perp}$",
            color=C_ORTHO, alpha=0.45, fontsize=10, ha="left", va="center",
            fontweight="bold")

    par, perp = A_PAR * DX, A_PERP * DY
    X0, X1 = 3.30, 4.70                    # components, then the resultant

    varrow(ax, (X0, 0), (X0 + par, 0), C_DRIFT, lw=1.5, head_length=4,
           head_width=3.0)
    varrow(ax, (X0 + par, 0), (X0 + par, perp), C_ORTHO, lw=1.5)
    sz = 0.11
    ax.plot([X0 + par - sz, X0 + par - sz, X0 + par], [0, sz, sz],
            color=C_MUTE, lw=0.7, zorder=2)

    ax.text(X0 + par, perp + 0.14,
            r"$\alpha_\perp \boldsymbol{\delta}_t^{\perp}$",
            color=C_ORTHO, fontsize=10.5, ha="center", va="bottom",
            fontweight="bold")
    # the reweighted drift component is too short to letter directly
    ax.annotate(r"$\alpha_\parallel \boldsymbol{\delta}_t^{\parallel}$",
                xy=(X0 + par * 0.55, -0.04),
                xytext=(X0 - 0.62, -0.45), color=C_DRIFT, fontsize=10.5,
                ha="center", va="center", fontweight="bold",
                arrowprops=dict(arrowstyle="-", lw=0.7, color=C_DRIFT,
                                alpha=0.75, shrinkA=3, shrinkB=1.5))

    ax.text(0.5 * (X0 + par + X1), perp * 0.45, r"$=$", fontsize=14,
            color=C_MUTE, ha="center", va="center")

    varrow(ax, (X1, 0), (X1 + par, perp), C_FINAL, lw=1.9)
    ax.text(X1 + par * 0.5, perp + 0.14, r"$\tilde{\boldsymbol{\delta}}_t$",
            color=C_FINAL, fontsize=12, va="bottom", ha="center",
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
    fig = plt.figure(figsize=(12.8, 2.78))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.545, 2.20], wspace=0.05)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    panel_geometry(axes[0])
    panel_reweight(axes[1])
    panel_pipeline(axes[2])

    # Panel titles sit in figure coordinates because the equal-aspect vector
    # panels letterbox their content, so an axes-relative title drifts away
    # from what it labels. Centred on the panel and nudged left by the same
    # amount everywhere.
    fig.canvas.draw()
    titles = ["(a) Decompose",
              r"(b) Reweight  ($\alpha_\perp \gg \alpha_\parallel$)",
              "(c) Pipeline"]
    for ax, t in zip(axes, titles):
        box = ax.get_position()
        fig.text(0.5 * (box.x0 + box.x1) - 0.013, 0.985, t, fontsize=10,
                 fontweight="bold", color=C_INK, ha="center", va="top")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "method_diagram.pdf")
    fig.savefig(OUT_DIR / "method_diagram.png", dpi=200)
    print("Saved", OUT_DIR / "method_diagram.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
