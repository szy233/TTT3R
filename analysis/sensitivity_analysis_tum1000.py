from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt

TAUS = [0.5, 1.0, 2.0, 4.0]
CUTOFFS = [2, 4, 8]

RESULTS_DIR = Path("eval_results/relpose/sensitivity/tum")
OUT_DIR = Path("analysis_results/sensitivity_tum1000")
OUT_DIR.mkdir(parents=True, exist_ok=True)

REF_CUT3R = 0.1641
REF_TTT3R = 0.1043
REF_TTT3R_JOINT = 0.0589

def parse_metric_file(path: Path):
    if not path.exists():
        return None
    text = path.read_text(errors="ignore")

    patterns = [
        r"ATE mean[:\s]+([0-9.eE+-]+)",
        r"ATE median[:\s]+([0-9.eE+-]+)",
        r"ATE \(mean\)[:\s]+([0-9.eE+-]+)",
        r"ATE \(median\)[:\s]+([0-9.eE+-]+)",
    ]

    nums = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    if len(nums) >= 2:
        mean_v = float(nums[0])
        median_v = float(nums[1])
        return mean_v, median_v

    vals = []
    for p in patterns:
        m = re.search(p, text)
        if m:
            vals.append(float(m.group(1)))
    if len(vals) >= 2:
        return vals[0], vals[1]
    return None

mean_grid = np.full((len(TAUS), len(CUTOFFS)), np.nan)
median_grid = np.full((len(TAUS), len(CUTOFFS)), np.nan)

print(f"Loading results from {RESULTS_DIR.resolve()}")
print("\n--- TUM relpose ---")

for i, tau in enumerate(TAUS):
    for j, cutoff in enumerate(CUTOFFS):
        tag = f"ttt3r_joint_tau{tau}_c{cutoff}"
        metric_path = RESULTS_DIR / tag / "rgbd_dataset_freiburg3_long_office_household_eval_metric.txt"
        parsed = parse_metric_file(metric_path)
        if parsed is None:
            print(f"  [missing] tum_s1_1000/{tag}")
            continue
        mean_grid[i, j], median_grid[i, j] = parsed

complete = np.sum(~np.isnan(median_grid))
print(f"  TUM: {complete}/12 configs complete")

def fmt_cell(v, ref):
    if np.isnan(v):
        return "  ---   "
    rel = (v - ref) / ref * 100.0
    return f"{v:.4f}({rel:+.1f}%)"

lines = []
lines.append("=" * 68)
lines.append("  TUM ATE  —  ATE median (m)")
lines.append(f"  Reference: cut3r={REF_CUT3R:.4f}  ttt3r={REF_TTT3R:.4f}  ttt3r_joint={REF_TTT3R_JOINT:.4f}")
lines.append("=" * 68)
header = "   τ/c   | " + " | ".join(f"  c={c}  " for c in CUTOFFS)
lines.append(header)
lines.append("  ---------+---------+---------+--------")
for i, tau in enumerate(TAUS):
    row = [fmt_cell(median_grid[i, j], REF_TTT3R_JOINT) for j in range(len(CUTOFFS))]
    lines.append(f"  τ={tau:<4} | " + " | ".join(row))

best_idx = np.unravel_index(np.nanargmin(median_grid), median_grid.shape) if np.any(~np.isnan(median_grid)) else None
if best_idx is not None:
    best_tau = TAUS[best_idx[0]]
    best_cutoff = CUTOFFS[best_idx[1]]
    best_val = median_grid[best_idx]
    lines.append("")
    lines.append("--- Best config ---")
    lines.append(f"  TUM: τ={best_tau}, c={best_cutoff}  →  {best_val:.4f}")

table_txt = OUT_DIR / "sensitivity_table.txt"
table_txt.write_text("\n".join(lines))
print("\n".join(lines))
print(f"\n[save] {table_txt.resolve()}")

latex_lines = []
latex_lines.append("\\begin{tabular}{c|ccc}")
latex_lines.append("$\\tau$ / cutoff & 2 & 4 & 8 \\\\")
latex_lines.append("\\hline")
for i, tau in enumerate(TAUS):
    row = []
    for j in range(len(CUTOFFS)):
        v = median_grid[i, j]
        row.append("--" if np.isnan(v) else f"{v:.4f}")
    latex_lines.append(f"{tau} & " + " & ".join(row) + " \\\\")
latex_lines.append("\\end{tabular}")

table_tex = OUT_DIR / "sensitivity_table.tex"
table_tex.write_text("\n".join(latex_lines))
print(f"[save] {table_tex.resolve()}")

plt.figure(figsize=(5, 4))
im = plt.imshow(median_grid, aspect="auto")
plt.xticks(range(len(CUTOFFS)), [str(c) for c in CUTOFFS])
plt.yticks(range(len(TAUS)), [str(t) for t in TAUS])
plt.xlabel("freq_cutoff")
plt.ylabel("tau")
plt.title("TUM ATE median")
plt.colorbar(im)
if best_idx is not None:
    plt.scatter(best_idx[1], best_idx[0], marker="s", s=120, facecolors="none", edgecolors="black", linewidths=2)
plt.tight_layout()
heatmap_pdf = OUT_DIR / "sensitivity_heatmap.pdf"
plt.savefig(heatmap_pdf)
plt.close()
print(f"[save] {heatmap_pdf.resolve()}")

plt.figure(figsize=(6, 4))
for j, cutoff in enumerate(CUTOFFS):
    plt.plot(TAUS, median_grid[:, j], marker="o", label=f"cutoff={cutoff}")
plt.axhline(REF_CUT3R, linestyle="--", label="cut3r")
plt.axhline(REF_TTT3R, linestyle="--", label="ttt3r")
plt.axhline(REF_TTT3R_JOINT, linestyle="--", label="ttt3r_joint")
plt.xlabel("tau")
plt.ylabel("ATE median")
plt.title("TUM sensitivity curves")
plt.legend()
plt.tight_layout()
curves_pdf = OUT_DIR / "sensitivity_curves.pdf"
plt.savefig(curves_pdf)
plt.close()
print(f"[save] {curves_pdf.resolve()}")

print(f"\n[done] All outputs in {OUT_DIR.resolve()}")