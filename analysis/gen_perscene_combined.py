"""Combined per-scene figure for analysis 3.2 and 3.3.

Two-panel scatter on ScanNet:
  (a) Gate variance vs ATE improvement -- no signal (3.2)
  (b) Drift energy vs ATE improvement -- significant negative correlation (3.3)
      blue = DDD3R helps, red = DDD3R hurts, gray squares = constant baseline.

Layout: panel (b) gets ~1.5x the width of (a) so it has room for legend +
all 90 points without being squished.

Sign convention: positive y = ATE improvement (lower ATE is better).
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from scipy import stats

BASE = Path("/home/szy/research/TTT3R")
OUT_DIR = BASE / "paper" / "fig"
A2 = BASE / "analysis_results" / "a1a2_dynamics"
A4_SUMMARY = BASE / "analysis_results" / "a4_delta_direction" / "a4_summary.txt"
RELPOSE_DIR = BASE / "eval_results" / "relpose" / "scannet_s3_90_first"

C_HELP  = "#3182BD"
C_HURT  = "#E74C3C"
C_CONST = "#AAAAAA"

rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def parse_ate(filepath):
    try:
        with open(filepath) as f:
            txt = f.read()
        m = re.search(r"APE w\.r\.t\. translation.*?mean\s+([\d.]+)", txt, re.DOTALL)
        return float(m.group(1)) if m else None
    except FileNotFoundError:
        return None


def collect_ates(method_subdir):
    out = {}
    d = RELPOSE_DIR / method_subdir
    if not d.exists():
        return out
    for f in d.iterdir():
        if f.name.endswith("_eval_metric.txt"):
            scene = f.name.replace("_eval_metric.txt", "")
            ate = parse_ate(f)
            if ate is not None:
                out[scene] = ate
    return out


def panel_gate_variance(ax):
    d = np.load(A2 / "a2_scannet_data.npz")
    var = d["cos_variances"]
    imp = d["improvements"]
    r = float(d["pearson_r"])
    p = float(d["pearson_p"])
    n = len(imp)
    n_help = int((imp > 0).sum())

    ax.axhline(0, color="#CCC", lw=0.5, zorder=0)
    ax.scatter(var, imp, s=22, alpha=0.65, c=C_HELP, edgecolor="white", lw=0.3,
               zorder=2)
    if len(var) >= 3:
        s, i, _, _, _ = stats.linregress(var, imp)
        xs = np.linspace(var.min(), var.max(), 100)
        ax.plot(xs, s * xs + i, "--", color=C_HELP, lw=0.8, alpha=0.55,
                zorder=1)

    ax.set_title("(a) Gate variance", fontsize=9, loc="left", fontweight="bold", pad=3)
    ax.set_xlabel(r"$\mathrm{Var}(\cos(\boldsymbol{\delta}_t, \boldsymbol{\delta}_{t-1}))$")
    ax.set_ylabel("ATE improvement (%)")
    # Stats annotation, top-left
    ax.text(0.04, 0.93,
            rf"$r{{=}}{r:+.2f}$, $p{{=}}{p:.2f}$ (n.s.)",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=7, color="#222")
    ax.text(0.96, 0.05, rf"$n{{=}}{n}$ scenes; helped {n_help}/{n}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6.5, color="#666")
    ax.tick_params(width=0.5, length=2.5)


def panel_drift_energy(ax):
    drift_data = {}
    with open(A4_SUMMARY) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 5 and parts[0] == "scannet":
                drift_data[parts[1]] = float(parts[4])

    ate_cut3r = collect_ates("cut3r")
    ate_ortho = collect_ates("ttt3r_ortho")
    ate_const = collect_ates("ttt3r_random")

    common = sorted(set(drift_data) & set(ate_cut3r) & set(ate_ortho))
    de = np.array([drift_data[s] for s in common])
    base = np.array([ate_cut3r[s] for s in common])
    valid = base > 0.001
    de = de[valid]
    common = [s for s, v in zip(common, valid) if v]

    imp_ortho = np.array([(ate_cut3r[s] - ate_ortho[s]) / ate_cut3r[s] * 100
                          for s in common])
    imp_const = np.array([(ate_cut3r[s] - ate_const.get(s, np.nan)) / ate_cut3r[s] * 100
                          if s in ate_const else np.nan for s in common])

    ax.axhline(0, color="#CCC", lw=0.5, zorder=0)

    valid_c = ~np.isnan(imp_const)
    if valid_c.any():
        ax.scatter(de[valid_c], imp_const[valid_c], c=C_CONST, alpha=0.32, s=16,
                   edgecolors="none", marker="s",
                   label=r"Constant $\alpha{=}0.5$", zorder=1)

    helps = imp_ortho > 0
    ax.scatter(de[helps], imp_ortho[helps], c=C_HELP, alpha=0.75, s=22,
               edgecolors="white", lw=0.3, zorder=3,
               label="DDD3R helps")
    ax.scatter(de[~helps], imp_ortho[~helps], c=C_HURT, alpha=0.78, s=22,
               edgecolors="white", lw=0.3, zorder=3,
               label="DDD3R hurts")

    s, i, r, p, _ = stats.linregress(de, imp_ortho)
    xs = np.linspace(de.min(), de.max(), 100)
    ax.plot(xs, s * xs + i, "-", color=C_HELP, lw=1.1, alpha=0.85, zorder=2,
            label=rf"DDD3R fit ($r{{=}}{r:+.2f}$, $p{{=}}{p:.3f}$)")

    ax.set_title("(b) Drift energy", fontsize=9, loc="left", fontweight="bold", pad=3)
    ax.set_xlabel(r"Drift energy $\bar{e}$")
    ax.set_ylim(-100, 70)
    ax.legend(loc="upper right", framealpha=0.85, edgecolor="none",
              borderaxespad=0.3, handletextpad=0.45)
    ax.tick_params(width=0.5, length=2.5)


def main():
    # width_ratios so (b) gets ~1.4x the width of (a), giving the legend
    # and the wider data range room without crushing the points
    fig = plt.figure(figsize=(5.5, 1.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.45], wspace=0.30)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    panel_gate_variance(ax_a)
    panel_drift_energy(ax_b)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "perscene_combined.pdf")
    fig.savefig(OUT_DIR / "perscene_combined.png", dpi=200)
    print("Saved perscene_combined.pdf/.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
