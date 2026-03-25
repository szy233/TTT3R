"""
Hyperparameter Sensitivity Analysis — τ × freq_cutoff
======================================================

Reads results produced by eval/run_sensitivity_eval.sh and generates:
  - sensitivity_heatmap.pdf  — 2×2 ATE heatmaps (ScanNet + TUM, mean + median)
  - sensitivity_heatmap_depth.pdf — depth Abs Rel heatmaps (KITTI + Bonn + Sintel)
  - sensitivity_curves.pdf   — ATE vs τ / vs cutoff line plots with reference lines
  - sensitivity_table.txt    — full text summary + LaTeX tables

Expected directory layout (under eval_results/):
  sensitivity/relpose/scannet_s3_1000/tau{τ}_c{cutoff}/_error_log.txt
  sensitivity/relpose/tum_s1_1000/tau{τ}_c{cutoff}/_error_log.txt
  sensitivity/video_depth/kitti_s1_500/tau{τ}_c{cutoff}/result_scale.json
  sensitivity/video_depth/bonn_s1_500/tau{τ}_c{cutoff}/result_scale.json
  sensitivity/video_depth/sintel/tau{τ}_c{cutoff}/result_scale.json
"""

import os
import re
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Grid ─────────────────────────────────────────────────────────────────────
TAUS    = [0.5, 1.0, 2.0, 4.0]
CUTOFFS = [2, 4, 8]

BASE_DIR    = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "eval_results"
OUT_DIR     = BASE_DIR / "analysis_results" / "sensitivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Reference values from formal eval (CLAUDE.md 2026-03-24/25)
REF_RELPOSE = {
    "scannet": {"cut3r": 0.6713, "ttt3r": 0.3519, "ttt3r_joint": 0.2143},
    "tum":     {"cut3r": 0.1641, "ttt3r": 0.1043, "ttt3r_joint": 0.0589},
}
REF_DEPTH = {
    "kitti":  {"cut3r": 0.1515, "ttt3r": 0.1319, "ttt3r_joint": 0.1344},
    "bonn":   {"cut3r": 0.0990, "ttt3r": 0.0997, "ttt3r_joint": 0.0941},
    "sintel": {"cut3r": 1.0217, "ttt3r": 0.9776, "ttt3r_joint": 0.9173},
}


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_error_log(log_path: Path):
    """
    Parse _error_log.txt from eval/relpose/launch.py (or video_depth/launch.py).
    Returns dict with ate_median, ate_mean, rpe_t_median, rpe_r_median, n_seq.
    Returns None if file missing or no valid sequences found.
    """
    if not log_path.is_file():
        return None

    pattern = re.compile(
        r"\|\s*ATE:\s*([\d.]+),\s*RPE trans:\s*([\d.]+),\s*RPE rot:\s*([\d.]+)"
    )
    ates, rpe_ts, rpe_rs = [], [], []
    with open(log_path) as f:
        for line in f:
            # Skip the "Average ATE: …" summary line
            if line.strip().startswith("Average"):
                continue
            m = pattern.search(line)
            if m:
                try:
                    ates.append(float(m.group(1)))
                    rpe_ts.append(float(m.group(2)))
                    rpe_rs.append(float(m.group(3)))
                except ValueError:
                    pass

    if not ates:
        return None

    return {
        "ate_median":   float(np.median(ates)),
        "ate_mean":     float(np.mean(ates)),
        "rpe_t_median": float(np.median(rpe_ts)),
        "rpe_r_median": float(np.median(rpe_rs)),
        "n_seq":        len(ates),
    }


def parse_depth_json(json_path: Path):
    """Parse result_scale.json; returns dict or None."""
    if not json_path.is_file():
        return None
    try:
        with open(json_path) as f:
            return json.load(f)
    except Exception:
        return None


# ── Grid loading ──────────────────────────────────────────────────────────────

def load_relpose_grid(dataset_dir_name: str):
    """
    Returns (mean_grid, median_grid) each shaped (len(TAUS), len(CUTOFFS)).
    dataset_dir_name: e.g. "scannet_s3_1000" or "tum_s1_1000"
    """
    mean_g   = np.full((len(TAUS), len(CUTOFFS)), np.nan)
    median_g = np.full((len(TAUS), len(CUTOFFS)), np.nan)
    base = RESULTS_DIR / "sensitivity" / "tum"

    for i, tau in enumerate(TAUS):
        for j, cutoff in enumerate(CUTOFFS):
            tag  = tag = f"ttt3r_joint_tau{tau}_c{cutoff}"
            log  = base / tag / "_error_log.txt"
            r    = parse_error_log(log)
            if r is None:
                print(f"  [missing] {dataset_dir_name}/{tag}")
                continue
            mean_g[i, j]   = r["ate_mean"]
            median_g[i, j] = r["ate_median"]
            print(f"  {dataset_dir_name}/{tag}: n={r['n_seq']}, "
                  f"median={r['ate_median']:.4f}, mean={r['ate_mean']:.4f}")

    return mean_g, median_g


def load_depth_grid(dataset_dir_name: str, metric_key: str = "Abs Rel"):
    """
    Returns value_grid shaped (len(TAUS), len(CUTOFFS)).
    dataset_dir_name: e.g. "kitti_s1_500", "bonn_s1_500", "sintel"
    """
    grid = np.full((len(TAUS), len(CUTOFFS)), np.nan)
    base = RESULTS_DIR / "sensitivity" / "video_depth" / dataset_dir_name

    for i, tau in enumerate(TAUS):
        for j, cutoff in enumerate(CUTOFFS):
            tag   = f"tau{tau}_c{cutoff}"
            jpath = base / tag / "result_scale.json"
            r     = parse_depth_json(jpath)
            if r is None:
                print(f"  [missing] {dataset_dir_name}/{tag}")
                continue
            if metric_key not in r:
                print(f"  [no key '{metric_key}'] {dataset_dir_name}/{tag}")
                continue
            grid[i, j] = r[metric_key]
            print(f"  {dataset_dir_name}/{tag}: {metric_key}={r[metric_key]:.4f}")

    return grid


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _mark_best(ax, grid):
    if not np.all(np.isnan(grid)):
        bi, bj = np.unravel_index(np.nanargmin(grid), grid.shape)
        ax.add_patch(plt.Rectangle(
            (bj - 0.5, bi - 0.5), 1, 1,
            fill=False, edgecolor="navy", linewidth=2.5
        ))


def plot_heatmap(ax, grid, title, colorbar_label="ATE median (m)",
                 ref_value=None, cmap="RdYlGn_r"):
    valid = grid[~np.isnan(grid)]
    if len(valid) == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, color="gray", fontsize=12)
        ax.set_title(f"{title}\n(no data)")
        return

    vmin, vmax = valid.min() * 0.98, valid.max() * 1.02
    im = ax.imshow(grid, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
    plt.colorbar(im, ax=ax, shrink=0.85, label=colorbar_label)

    ax.set_xticks(range(len(CUTOFFS)))
    ax.set_xticklabels([f"c={c}" for c in CUTOFFS])
    ax.set_yticks(range(len(TAUS)))
    ax.set_yticklabels([f"τ={t}" for t in TAUS])
    ax.set_xlabel("freq_cutoff")
    ax.set_ylabel("geo_gate_tau")
    ax.set_title(title)

    for i in range(len(TAUS)):
        for j in range(len(CUTOFFS)):
            v = grid[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=9, color="gray")
                continue
            txt = f"{v:.4f}"
            if ref_value is not None:
                d = (v - ref_value) / (ref_value + 1e-9) * 100
                txt += f"\n({d:+.1f}%)"
            color = "white" if v > (vmin + vmax) / 2 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5, color=color)

    _mark_best(ax, grid)


def plot_curves(ax, grid, title, ylabel, ref_lines=None, xscale="log"):
    """ATE/metric vs τ, one line per cutoff."""
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(CUTOFFS)))
    for j, (c, color) in enumerate(zip(CUTOFFS, colors)):
        col  = grid[:, j]
        mask = ~np.isnan(col)
        if mask.sum() < 2:
            continue
        ax.plot(np.array(TAUS)[mask], col[mask], "o-",
                color=color, label=f"c={c}", linewidth=1.8, markersize=6)

    if ref_lines:
        for key, color, ls, lw in [
            ("cut3r",      "#999999", ":",  1.2),
            ("ttt3r",      "#4488cc", "--", 1.2),
            ("ttt3r_joint","#cc4444", "-.", 1.2),
        ]:
            if key in ref_lines:
                ax.axhline(ref_lines[key], color=color, linestyle=ls,
                           linewidth=lw, label=key, alpha=0.8)

    if xscale == "log":
        ax.set_xscale("log")
    ax.set_xticks(TAUS)
    ax.set_xticklabels([str(t) for t in TAUS])
    ax.set_xlabel("geo_gate_tau (τ)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_curves_vs_cutoff(ax, grid, title, ylabel, ref_lines=None):
    """ATE/metric vs cutoff, one line per τ."""
    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(TAUS)))
    for i, (tau, color) in enumerate(zip(TAUS, colors)):
        row  = grid[i, :]
        mask = ~np.isnan(row)
        if mask.sum() < 2:
            continue
        ax.plot(np.array(CUTOFFS)[mask], row[mask], "s-",
                color=color, label=f"τ={tau}", linewidth=1.8, markersize=6)

    if ref_lines:
        for key, color, ls, lw in [
            ("cut3r",      "#999999", ":",  1.2),
            ("ttt3r",      "#4488cc", "--", 1.2),
            ("ttt3r_joint","#cc4444", "-.", 1.2),
        ]:
            if key in ref_lines:
                ax.axhline(ref_lines[key], color=color, linestyle=ls,
                           linewidth=lw, label=key, alpha=0.8)

    ax.set_xticks(CUTOFFS)
    ax.set_xticklabels([f"H//{c}" for c in CUTOFFS])
    ax.set_xlabel("freq_cutoff (H//c bandwidth)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


# ── Table helpers ─────────────────────────────────────────────────────────────

def make_text_table(grid, dataset_label, ref_map, metric_label="ATE median (m)"):
    """Single-dataset text table (rows=τ, cols=cutoff)."""
    lines = []
    lines.append(f"\n{'='*68}")
    lines.append(f"  {dataset_label}  —  {metric_label}")
    ref_str = "  ".join(f"{k}={v:.4f}" for k, v in ref_map.items())
    lines.append(f"  Reference: {ref_str}")
    lines.append(f"{'='*68}")

    header = "   τ/c   | " + " | ".join(f"  c={c}  " for c in CUTOFFS)
    lines.append(header)
    lines.append("  " + "-" * 8 + "-+-" + "-+-".join(["-" * 7] * len(CUTOFFS)))

    for i, tau in enumerate(TAUS):
        cells = []
        for j in range(len(CUTOFFS)):
            v = grid[i, j]
            if np.isnan(v):
                cells.append("  ---   ")
            else:
                ref_base = ref_map.get("ttt3r_joint") or list(ref_map.values())[-1]
                pct = (v / ref_base - 1) * 100 if ref_base else 0
                cells.append(f"{v:.4f}({pct:+.1f}%)")
        lines.append(f"  τ={tau:<5} | " + " | ".join(cells))
    lines.append("")
    return "\n".join(lines)


def make_latex_table(sn_grid, tum_grid, kitti_grid, bonn_grid, sintel_grid):
    """Combined LaTeX sensitivity table for all 5 datasets."""
    def _cell(v, best_mask):
        if np.isnan(v):
            return "--"
        s = f"{v:.4f}"
        return rf"\mathbf{{{s}}}" if best_mask else s

    def _best_mask(grid):
        mask = np.zeros_like(grid, dtype=bool)
        if not np.all(np.isnan(grid)):
            bi, bj = np.unravel_index(np.nanargmin(grid), grid.shape)
            mask[bi, bj] = True
        return mask

    bsn = _best_mask(sn_grid); btum = _best_mask(tum_grid)
    bkit = _best_mask(kitti_grid); bbon = _best_mask(bonn_grid); bsin = _best_mask(sintel_grid)

    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Sensitivity of \texttt{ttt3r\_joint} to $\tau$ and \texttt{freq\_cutoff}."
        r" Metrics: ATE$\downarrow$ (m, median) for relpose; Abs\,Rel$\downarrow$ for depth."
        r" \textbf{Bold} = best per column group. $\dagger$ marks the recommended config.}",
        r"  \label{tab:sensitivity}",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \small",
        r"  \begin{tabular}{cc|ccc|ccc|ccc|ccc|ccc}",
        r"    \toprule",
        r"    & & \multicolumn{3}{c|}{ScanNet ATE$\downarrow$} & \multicolumn{3}{c|}{TUM ATE$\downarrow$}"
        r" & \multicolumn{3}{c|}{KITTI Abs\,Rel$\downarrow$} & \multicolumn{3}{c|}{Bonn Abs\,Rel$\downarrow$}"
        r" & \multicolumn{3}{c}{Sintel Abs\,Rel$\downarrow$} \\",
        r"    $\tau$ & $c$ & $H/2$ & $H/4$ & $H/8$ & $H/2$ & $H/4$ & $H/8$"
        r" & $H/2$ & $H/4$ & $H/8$ & $H/2$ & $H/4$ & $H/8$ & $H/2$ & $H/4$ & $H/8$ \\",
        r"    \midrule",
        r"    \multicolumn{2}{l|}{\textit{Reference}} \\",
    ]

    # Reference block
    for cfg, label in [("cut3r", r"\texttt{cut3r}"),
                        ("ttt3r", r"\texttt{ttt3r}"),
                        ("ttt3r\_joint", r"\texttt{ttt3r\_joint}")]:
        sn_v  = REF_RELPOSE["scannet"].get(cfg, float("nan"))
        tum_v = REF_RELPOSE["tum"].get(cfg, float("nan"))
        kit_v = REF_DEPTH["kitti"].get(cfg, float("nan"))
        bon_v = REF_DEPTH["bonn"].get(cfg, float("nan"))
        sin_v = REF_DEPTH["sintel"].get(cfg, float("nan"))
        fmt   = lambda v: f"{v:.4f}" if not np.isnan(v) else "--"

        lines.append(
            f"    \\multicolumn{{2}}{{l|}}{{{label}}} & "
            + " & ".join([fmt(sn_v)] * 3) + " & "
            + " & ".join([fmt(tum_v)] * 3) + " & "
            + " & ".join([fmt(kit_v)] * 3) + " & "
            + " & ".join([fmt(bon_v)] * 3) + " & "
            + " & ".join([fmt(sin_v)] * 3) + r" \\"
        )

    lines.append(r"    \midrule")
    lines.append(r"    \multicolumn{2}{l|}{\textit{Grid (ttt3r\_joint)}} \\")

    for i, tau in enumerate(TAUS):
        row_cells = []
        for j in range(len(CUTOFFS)):
            row_cells.append(_cell(sn_grid[i,j],    bsn[i,j]))
        for j in range(len(CUTOFFS)):
            row_cells.append(_cell(tum_grid[i,j],   btum[i,j]))
        for j in range(len(CUTOFFS)):
            row_cells.append(_cell(kitti_grid[i,j], bkit[i,j]))
        for j in range(len(CUTOFFS)):
            row_cells.append(_cell(bonn_grid[i,j],  bbon[i,j]))
        for j in range(len(CUTOFFS)):
            row_cells.append(_cell(sintel_grid[i,j],bsin[i,j]))

        # Mark recommended config (τ=2, c=4)
        tau_dagger = rf"{tau}$^\dagger$" if (tau == 2.0) else str(tau)
        lines.append(f"    {tau_dagger} & {CUTOFFS[0]}--{CUTOFFS[-1]} & "
                     + " & ".join(row_cells) + r" \\")

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \\[0.4em]{\footnotesize $\dagger$ recommended: $\tau=2$, $c=4$}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def make_compact_latex(sn_grid, tum_grid):
    """Compact τ × cutoff table for relpose only (fits in a single column)."""
    def _cell(v, is_best):
        if np.isnan(v):
            return "--"
        s = f"{v:.4f}"
        return rf"\textbf{{{s}}}" if is_best else s

    lines = [
        r"\begin{table}[h]",
        r"  \centering",
        r"  \caption{Sensitivity of \texttt{ttt3r\_joint} to $\tau$ and freq\_cutoff."
        r" ATE$\downarrow$ (median, m). \textbf{Bold} = row-best.}",
        r"  \label{tab:sensitivity_compact}",
        r"  \small",
        r"  \begin{tabular}{cc|" + "c" * len(CUTOFFS) + "|" + "c" * len(CUTOFFS) + "}",
        r"    \toprule",
        r"    & & \multicolumn{" + str(len(CUTOFFS)) + r"}{c|}{ScanNet ATE$\downarrow$}"
        + r" & \multicolumn{" + str(len(CUTOFFS)) + r"}{c}{TUM ATE$\downarrow$} \\",
        r"    $\tau$ & & " + " & ".join(f"$H/{c}$" for c in CUTOFFS)
        + " & " + " & ".join(f"$H/{c}$" for c in CUTOFFS) + r" \\",
        r"    \midrule",
    ]

    # Reference rows
    for cfg, label in [("cut3r", r"\texttt{cut3r}"), ("ttt3r", r"\texttt{ttt3r}")]:
        sn_v  = REF_RELPOSE["scannet"].get(cfg, float("nan"))
        tum_v = REF_RELPOSE["tum"].get(cfg, float("nan"))
        fmt = lambda v: f"{v:.4f}" if not np.isnan(v) else "--"
        lines.append(
            f"    \\multicolumn{{2}}{{l|}}{{{label}}} & "
            + " & ".join([fmt(sn_v)] * len(CUTOFFS)) + " & "
            + " & ".join([fmt(tum_v)] * len(CUTOFFS)) + r" \\"
        )
    lines.append(r"    \midrule")

    for i, tau in enumerate(TAUS):
        sn_row  = sn_grid[i, :]
        tum_row = tum_grid[i, :]

        sn_best  = int(np.nanargmin(sn_row))  if not np.all(np.isnan(sn_row))  else -1
        tum_best = int(np.nanargmin(tum_row)) if not np.all(np.isnan(tum_row)) else -1

        sn_cells  = [_cell(sn_row[j],  j == sn_best)  for j in range(len(CUTOFFS))]
        tum_cells = [_cell(tum_row[j], j == tum_best) for j in range(len(CUTOFFS))]

        tau_mark = rf"{tau}$^\dagger$" if tau == 2.0 else str(tau)
        lines.append(f"    {tau_mark} & & " + " & ".join(sn_cells)
                     + " & " + " & ".join(tum_cells) + r" \\")

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \\[0.4em]{\footnotesize $\dagger$ recommended config ($\tau\!=\!2$).}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\nLoading results from {RESULTS_DIR}/sensitivity/")

    print("\n--- ScanNet relpose ---")
    sn_mean, sn_median = load_relpose_grid("scannet_s3_1000")

    print("\n--- TUM relpose ---")
    tum_mean, tum_median = load_relpose_grid("tum_s1_50")

    print("\n--- KITTI depth ---")
    kitti_grid = load_depth_grid("kitti_s1_500")

    print("\n--- Bonn depth ---")
    bonn_grid = load_depth_grid("bonn_s1_500")

    print("\n--- Sintel depth ---")
    sintel_grid = load_depth_grid("sintel")

    # ── Coverage ──
    total = len(TAUS) * len(CUTOFFS)
    for name, g in [("ScanNet", sn_median), ("TUM", tum_median),
                    ("KITTI",  kitti_grid), ("Bonn", bonn_grid), ("Sintel", sintel_grid)]:
        n = int(np.sum(~np.isnan(g)))
        print(f"  {name:10}: {n}/{total} configs complete")

    # ── Text tables ──────────────────────────────────────────────────────────
    table_text = ""
    table_text += make_text_table(sn_median,  "ScanNet ATE",    REF_RELPOSE["scannet"])
    table_text += make_text_table(tum_median, "TUM ATE",        REF_RELPOSE["tum"])
    table_text += make_text_table(kitti_grid, "KITTI Abs Rel",  REF_DEPTH["kitti"])
    table_text += make_text_table(bonn_grid,  "Bonn Abs Rel",   REF_DEPTH["bonn"])
    table_text += make_text_table(sintel_grid,"Sintel Abs Rel", REF_DEPTH["sintel"])

    # Best config summary
    table_text += "\n--- Best configs ---\n"
    for name, grid in [("ScanNet", sn_median), ("TUM", tum_median),
                       ("KITTI",   kitti_grid), ("Bonn", bonn_grid), ("Sintel", sintel_grid)]:
        if not np.all(np.isnan(grid)):
            bi, bj = np.unravel_index(np.nanargmin(grid), grid.shape)
            table_text += f"  {name:10}: τ={TAUS[bi]}, c={CUTOFFS[bj]}  →  {grid[bi,bj]:.4f}\n"

    table_path = OUT_DIR / "sensitivity_table.txt"
    table_path.write_text(table_text)
    print(f"\n[save] {table_path}")
    print(table_text)

    # ── LaTeX tables ─────────────────────────────────────────────────────────
    latex_full    = make_latex_table(sn_median, tum_median, kitti_grid, bonn_grid, sintel_grid)
    latex_compact = make_compact_latex(sn_median, tum_median)

    latex_path = OUT_DIR / "sensitivity_table.tex"
    with open(latex_path, "w") as f:
        f.write("% Full sensitivity table (all 5 datasets)\n")
        f.write(latex_full + "\n\n")
        f.write("% Compact relpose-only table\n")
        f.write(latex_compact + "\n")
    print(f"[save] {latex_path}")

    # ── Figure 1: Relpose heatmaps (2×2) ─────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(r"ttt3r\_joint Sensitivity — ATE$\downarrow$ (Median, m)",
                 fontsize=13, fontweight="bold")

    plot_heatmap(axes[0, 0], sn_median,  "ScanNet — ATE median",
                 ref_value=REF_RELPOSE["scannet"]["ttt3r_joint"])
    plot_heatmap(axes[0, 1], sn_mean,    "ScanNet — ATE mean",
                 ref_value=REF_RELPOSE["scannet"]["ttt3r_joint"])
    plot_heatmap(axes[1, 0], tum_median, "TUM — ATE median",
                 ref_value=REF_RELPOSE["tum"]["ttt3r_joint"])
    plot_heatmap(axes[1, 1], tum_mean,   "TUM — ATE mean",
                 ref_value=REF_RELPOSE["tum"]["ttt3r_joint"])

    fig.text(0.5, -0.01,
             r"Cells: value ($\Delta$% vs ttt3r\_joint at $\tau$=2, c=4). "
             r"$\blacksquare$ navy = best in grid.",
             ha="center", fontsize=9, color="dimgray")
    plt.tight_layout()
    hp = OUT_DIR / "sensitivity_heatmap.pdf"
    plt.savefig(hp, dpi=150, bbox_inches="tight")
    plt.savefig(hp.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[save] {hp}")

    # ── Figure 2: Depth heatmaps (3×1) ───────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    fig2.suptitle(r"ttt3r\_joint Sensitivity — Abs Rel$\downarrow$ (Video Depth)",
                  fontsize=13, fontweight="bold")

    for ax, grd, ds, ref_key in zip(
        axes2,
        [kitti_grid,  bonn_grid,   sintel_grid],
        ["KITTI",     "Bonn",      "Sintel"],
        ["kitti",     "bonn",      "sintel"],
    ):
        plot_heatmap(ax, grd, f"{ds} — Abs Rel",
                     colorbar_label="Abs Rel",
                     ref_value=REF_DEPTH[ref_key]["ttt3r_joint"],
                     cmap="RdYlGn_r")

    plt.tight_layout()
    hdp = OUT_DIR / "sensitivity_heatmap_depth.pdf"
    plt.savefig(hdp, dpi=150, bbox_inches="tight")
    plt.savefig(hdp.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[save] {hdp}")

    # ── Figure 3: Relpose sensitivity curves (2×2) ────────────────────────────
    fig3, axes3 = plt.subplots(2, 2, figsize=(13, 9))
    fig3.suptitle(r"ttt3r\_joint Sensitivity Curves — ATE Median$\downarrow$",
                  fontsize=13, fontweight="bold")

    plot_curves(axes3[0, 0], sn_median, "ScanNet: ATE vs τ", "ATE median (m)",
                ref_lines=REF_RELPOSE["scannet"])
    plot_curves_vs_cutoff(axes3[0, 1], sn_median, "ScanNet: ATE vs freq_cutoff", "ATE median (m)",
                          ref_lines=REF_RELPOSE["scannet"])
    plot_curves(axes3[1, 0], tum_median, "TUM: ATE vs τ", "ATE median (m)",
                ref_lines=REF_RELPOSE["tum"])
    plot_curves_vs_cutoff(axes3[1, 1], tum_median, "TUM: ATE vs freq_cutoff", "ATE median (m)",
                          ref_lines=REF_RELPOSE["tum"])

    plt.tight_layout()
    cp = OUT_DIR / "sensitivity_curves.pdf"
    plt.savefig(cp, dpi=150, bbox_inches="tight")
    plt.savefig(cp.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[save] {cp}")

    # ── Figure 4: Depth sensitivity curves (2×3) ─────────────────────────────
    fig4, axes4 = plt.subplots(2, 3, figsize=(15, 9))
    fig4.suptitle(r"ttt3r\_joint Sensitivity Curves — Abs Rel$\downarrow$ (Video Depth)",
                  fontsize=13, fontweight="bold")

    for col, (grd, ds, ref_key) in enumerate(zip(
        [kitti_grid,  bonn_grid,   sintel_grid],
        ["KITTI",     "Bonn",      "Sintel"],
        ["kitti",     "bonn",      "sintel"],
    )):
        plot_curves(axes4[0, col], grd, f"{ds}: Abs Rel vs τ", "Abs Rel",
                    ref_lines=REF_DEPTH[ref_key])
        plot_curves_vs_cutoff(axes4[1, col], grd, f"{ds}: Abs Rel vs cutoff", "Abs Rel",
                              ref_lines=REF_DEPTH[ref_key])

    plt.tight_layout()
    cdp = OUT_DIR / "sensitivity_curves_depth.pdf"
    plt.savefig(cdp, dpi=150, bbox_inches="tight")
    plt.savefig(cdp.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[save] {cdp}")

    print(f"\n[done] All outputs in {OUT_DIR}/")


if __name__ == "__main__":
    main()
