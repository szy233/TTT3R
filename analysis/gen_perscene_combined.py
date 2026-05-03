"""Combined per-scene figure for analysis 3.2 and 3.3.

Two-panel scatter on ScanNet:
  (a) Gate variance vs ATE improvement -- no signal (3.2)
  (b) Drift energy vs ATE improvement  -- significant positive correlation (3.3)

Shared y-axis (improvement %), illustrates the contrast: the gate's
own driving signal carries no predictive information, but drift
energy does.

Saves to paper/fig/perscene_combined.{pdf,png}.
"""
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

C_HELP = "#3182BD"   # blue: DDD3R helps (improvement < 0, lower ATE)
C_HURT = "#E74C3C"   # red:  DDD3R hurts (improvement > 0)
C_CONST = "#AAAAAA"  # gray: constant baseline

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
    "savefig.pad_inches": 0.03,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


import re


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


# ─── Panel (a): gate variance vs improvement (ScanNet, 65 scenes) ──────────
def panel_gate_variance(ax):
    d = np.load(A2 / "a2_scannet_data.npz")
    var = d["cos_variances"]
    imp = d["improvements"]   # already in %, brake vs random
    r = float(d["pearson_r"])
    p = float(d["pearson_p"])
    n = len(imp)
    n_help = int((imp > 0).sum())

    ax.axhline(0, color="#CCC", lw=0.5, zorder=0)
    ax.scatter(var, imp, s=22, alpha=0.6, c=C_HELP, edgecolor="white", lw=0.3)
    # Faint linear fit purely for visual reference
    if len(var) >= 3:
        s, i, _, _, _ = stats.linregress(var, imp)
        xs = np.linspace(var.min(), var.max(), 100)
        ax.plot(xs, s * xs + i, "--", color=C_HELP, lw=0.7, alpha=0.55)

    ax.set_title(rf"(a) Gate variance ($r{{=}}{r:+.2f}$, $p{{=}}{p:.2f}$)",
                 fontsize=8.5, loc="left", fontweight="bold", pad=3)
    ax.set_xlabel(r"$\mathrm{Var}(\cos(\boldsymbol{\delta}_t, \boldsymbol{\delta}_{t-1}))$")
    ax.text(0.97, 0.05, rf"helped {n_help}/{n}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6.5, color="#444")
    ax.tick_params(width=0.5, length=2.5)


# ─── Panel (b): drift energy vs improvement (ScanNet, ~90 scenes) ──────────
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

    # Sign convention: positive = improvement (lower ATE is better)
    imp_ortho = np.array([(ate_cut3r[s] - ate_ortho[s]) / ate_cut3r[s] * 100
                          for s in common])
    imp_const = np.array([(ate_cut3r[s] - ate_const.get(s, np.nan)) / ate_cut3r[s] * 100
                          if s in ate_const else np.nan for s in common])

    ax.axhline(0, color="#CCC", lw=0.5, zorder=0)

    # Constant baseline (background)
    valid_c = ~np.isnan(imp_const)
    if valid_c.any():
        ax.scatter(de[valid_c], imp_const[valid_c], c=C_CONST, alpha=0.32, s=14,
                   edgecolors="none", marker="s",
                   label=r"Constant $\alpha{=}0.5$", zorder=1)

    # DDD3R points: blue if helps (imp>0), red if hurts (imp<0)
    helps = imp_ortho > 0
    ax.scatter(de[helps], imp_ortho[helps], c=C_HELP, alpha=0.7, s=20,
               edgecolors="white", lw=0.3, zorder=3,
               label="DDD3R helps")
    ax.scatter(de[~helps], imp_ortho[~helps], c=C_HURT, alpha=0.75, s=20,
               edgecolors="white", lw=0.3, zorder=3,
               label="DDD3R hurts")

    # Fit on DDD3R; correlation sign should be NEGATIVE (more drift -> less help)
    s, i, r, p, _ = stats.linregress(de, imp_ortho)
    xs = np.linspace(de.min(), de.max(), 100)
    ax.plot(xs, s * xs + i, "-", color=C_HELP, lw=1.0, alpha=0.85, zorder=2)

    ax.set_title(rf"(b) Drift energy ($r{{=}}{r:+.2f}$, $p{{=}}{p:.3f}$)",
                 fontsize=8.5, loc="left", fontweight="bold", pad=3)
    ax.set_xlabel(r"Drift energy $\bar{e}$")
    ax.legend(loc="upper left", framealpha=0.85, edgecolor="none", borderaxespad=0.3)
    ax.tick_params(width=0.5, length=2.5)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.1))
    panel_gate_variance(axes[0])
    panel_drift_energy(axes[1])
    axes[0].set_ylabel("ATE improvement (%)")
    axes[1].set_ylabel("ATE improvement (%)")
    axes[1].set_ylim(-100, 60)
    fig.tight_layout(w_pad=1.4)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "perscene_combined.pdf")
    fig.savefig(OUT_DIR / "perscene_combined.png", dpi=200)
    print("Saved perscene_combined.pdf/.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
