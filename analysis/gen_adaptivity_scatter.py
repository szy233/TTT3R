"""Per-scene gate variance vs ATE improvement (analysis 3.2).

Renders a 2-panel scatter for fig:adaptivity using A2 aggregated data
on TUM (8 scenes) and ScanNet (65 scenes), showing that gate variance
does not predict per-scene improvement (near-zero correlations on both).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

BASE = Path("/home/szy/research/TTT3R")
OUT_DIR = BASE / "paper" / "fig"
DATA_DIR = BASE / "analysis_results" / "a1a2_dynamics"

rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def panel(ax, dataset, title, color):
    d = np.load(DATA_DIR / f"a2_{dataset}_data.npz")
    var = d["cos_variances"]
    imp = d["improvements"]
    r = float(d["pearson_r"])
    p = float(d["pearson_p"])
    n_pos = int((imp > 0).sum())
    n = len(imp)

    ax.axhline(0, color="#999", lw=0.5, zorder=0)
    ax.scatter(var, imp, s=28, alpha=0.75, c=color, edgecolor="white", lw=0.4)

    # Linear fit (purely visual; not significant on either)
    if len(var) >= 3:
        z = np.polyfit(var, imp, 1)
        xs = np.linspace(var.min(), var.max(), 100)
        ax.plot(xs, np.polyval(z, xs), ls="--", color=color, lw=0.9, alpha=0.7)

    ax.set_title(rf"{title} ($n{{=}}{n}$): $r{{=}}{r:+.3f}$, $p{{=}}{p:.2f}$",
                 fontsize=8.5, loc="left", fontweight="bold")
    ax.text(0.97, 0.05, rf"improved {n_pos}/{n}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, color="#444")
    ax.set_xlabel(r"Var($\cos(\boldsymbol{\delta}_t, \boldsymbol{\delta}_{t-1})$)")
    ax.tick_params(width=0.5, length=2.5)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.9))
    panel(axes[0], "tum",     "TUM",     "#3182BD")
    panel(axes[1], "scannet", "ScanNet", "#5BAA5B")
    axes[0].set_ylabel("ATE improvement (%)")
    fig.tight_layout(w_pad=1.6)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "adaptivity_scatter.pdf")
    fig.savefig(OUT_DIR / "adaptivity_scatter.png", dpi=200)
    print("Saved adaptivity_scatter.pdf/.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
