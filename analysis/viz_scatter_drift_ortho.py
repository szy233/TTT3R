"""Per-scene scatter plot: drift energy vs ortho improvement.

Shows that high drift energy scenes (ScanNet) tend to have WORSE ortho
improvement, supporting the narrative that drift in high-drift scenes
is useful refinement, not noise.

Usage:
    python analysis/viz_scatter_drift_ortho.py
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

EVAL_BASE = "eval_results/relpose/scannet_s3_90_first"
A4_DATA = "analysis_results/a4_delta_direction/a4_all_data.npz"
A4_SUMMARY = "analysis_results/a4_delta_direction/a4_summary.txt"
OUTPUT_DIR = "analysis_results/scatter_drift_ortho"

# Methods to compare
CONFIGS = {
    "cut3r": "cut3r",
    "ortho": "ttt3r_ortho",
    "random": "ttt3r_random",
    "brake": "ttt3r_momentum_inv_t1",
    "ttt3r": "ttt3r",
    "ortho_adaptive": "ttt3r_ortho_adaptive",
}


def parse_ate_from_metric(filepath):
    """Parse ATE mean from evo metric file."""
    with open(filepath) as f:
        txt = f.read()
    m = re.search(r"APE w\.r\.t\. translation.*?mean\s+([\d.]+)", txt, re.DOTALL)
    if m:
        return float(m.group(1))
    return None


def collect_per_scene_ate(config_dir):
    """Collect per-scene ATE from eval results."""
    results = {}
    if not os.path.exists(config_dir):
        return results
    for fname in os.listdir(config_dir):
        if fname.endswith("_eval_metric.txt"):
            scene = fname.replace("_eval_metric.txt", "")
            ate = parse_ate_from_metric(os.path.join(config_dir, fname))
            if ate is not None:
                results[scene] = ate
    return results


def parse_a4_summary():
    """Parse per-scene drift energy from A4 summary."""
    scenes = {}
    with open(A4_SUMMARY) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 5 and parts[0] == "scannet":
                scene = parts[1]
                drift_energy = float(parts[4])
                cos_mean = float(parts[2])
                scenes[scene] = {"drift_energy": drift_energy, "cos_mean": cos_mean}
    return scenes


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load drift energy data
    a4_scenes = parse_a4_summary()
    print(f"A4 data: {len(a4_scenes)} ScanNet scenes")

    # Collect per-scene ATE for all methods
    all_ates = {}
    for label, dirname in CONFIGS.items():
        config_dir = os.path.join(EVAL_BASE, dirname)
        ates = collect_per_scene_ate(config_dir)
        all_ates[label] = ates
        print(f"  {label}: {len(ates)} scenes")

    # Find common scenes across cut3r, ortho, and A4
    common = set(a4_scenes.keys())
    for label in ["cut3r", "ortho"]:
        common &= set(all_ates[label].keys())
    common = sorted(common)
    print(f"\nCommon scenes (cut3r ∩ ortho ∩ A4): {len(common)}")

    # Build arrays
    drift_energy = np.array([a4_scenes[s]["drift_energy"] for s in common])
    cos_mean = np.array([a4_scenes[s]["cos_mean"] for s in common])
    ate_cut3r = np.array([all_ates["cut3r"][s] for s in common])
    ate_ortho = np.array([all_ates["ortho"][s] for s in common])

    # Filter valid scenes (non-zero baseline)
    valid = ate_cut3r > 0.001
    drift_energy = drift_energy[valid]
    cos_mean = cos_mean[valid]
    ate_cut3r = ate_cut3r[valid]
    ate_ortho = ate_ortho[valid]
    common = [s for s, v in zip(common, valid) if v]
    print(f"Valid scenes: {len(common)}")

    # Compute ortho improvement (negative = ortho better)
    improvement = (ate_ortho - ate_cut3r) / ate_cut3r * 100  # percentage

    # Also compute for other methods
    method_improvements = {}
    for label in ["random", "brake", "ttt3r", "ortho_adaptive"]:
        if label in all_ates:
            imp = []
            for s, base in zip(common, ate_cut3r):
                if s in all_ates[label]:
                    imp.append((all_ates[label][s] - base) / base * 100)
                else:
                    imp.append(np.nan)
            method_improvements[label] = np.array(imp)

    # === Figure 1: Main scatter — drift energy vs ortho improvement ===
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Color by improvement direction
    colors = np.where(improvement < 0, '#2196F3', '#F44336')  # blue=improved, red=degraded
    ax.scatter(drift_energy, improvement, c=colors, alpha=0.6, s=40, edgecolors='white', linewidth=0.5)

    # Add trend line
    slope, intercept, r_value, p_value, std_err = stats.linregress(drift_energy, improvement)
    x_fit = np.linspace(drift_energy.min(), drift_energy.max(), 100)
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, 'k--', alpha=0.7, linewidth=1.5,
            label=f"r={r_value:.3f}, p={p_value:.3f}")

    ax.axhline(0, color='gray', linestyle='-', alpha=0.3, linewidth=1)
    ax.set_xlabel("Drift Energy (cos² EMA)", fontsize=12)
    ax.set_ylabel("Ortho vs CUT3R (% change in ATE)", fontsize=12)
    ax.set_title("Per-Scene: Drift Energy vs Ortho Improvement\n(ScanNet 90f)", fontsize=13)
    ax.legend(fontsize=10, loc='upper left')

    # Annotate counts
    n_improved = (improvement < 0).sum()
    n_degraded = (improvement > 0).sum()
    ax.text(0.98, 0.02, f"Improved: {n_improved}/{len(improvement)}\nDegraded: {n_degraded}/{len(improvement)}",
            transform=ax.transAxes, fontsize=10, ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "scatter_drift_vs_ortho.png"), dpi=200, bbox_inches='tight')
    fig.savefig(os.path.join(OUTPUT_DIR, "scatter_drift_vs_ortho.pdf"), bbox_inches='tight')
    print(f"\nSaved scatter_drift_vs_ortho.png/pdf")
    plt.close(fig)

    # === Figure 2: Multi-method comparison scatter ===
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    method_pairs = [
        ("ortho", improvement, "Ortho"),
        ("brake", method_improvements.get("brake"), "Brake"),
        ("random", method_improvements.get("random"), "Random (p=0.5)"),
    ]

    for ax, (label, imp, title) in zip(axes, method_pairs):
        if imp is None:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha='center')
            continue
        valid_imp = ~np.isnan(imp)
        de = drift_energy[valid_imp]
        im = imp[valid_imp]

        colors = np.where(im < 0, '#2196F3', '#F44336')
        ax.scatter(de, im, c=colors, alpha=0.6, s=30, edgecolors='white', linewidth=0.5)

        # Trend
        if len(de) > 5:
            s, i, r, p, _ = stats.linregress(de, im)
            x_f = np.linspace(de.min(), de.max(), 100)
            ax.plot(x_f, s * x_f + i, 'k--', alpha=0.7, linewidth=1.5,
                    label=f"r={r:.3f}, p={p:.3f}")

        ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
        ax.set_xlabel("Drift Energy", fontsize=11)
        ax.set_ylabel("% change in ATE vs CUT3R", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='upper left')

        n_imp = (im < 0).sum()
        ax.text(0.98, 0.02, f"{n_imp}/{len(im)} improved",
                transform=ax.transAxes, fontsize=9, ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.8))

    fig.suptitle("Per-Scene: Drift Energy vs Method Improvement (ScanNet 90f)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "scatter_multi_method.png"), dpi=200, bbox_inches='tight')
    fig.savefig(os.path.join(OUTPUT_DIR, "scatter_multi_method.pdf"), bbox_inches='tight')
    print("Saved scatter_multi_method.png/pdf")
    plt.close(fig)

    # === Figure 3: Drift energy distribution — improved vs degraded scenes ===
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    improved_de = drift_energy[improvement < 0]
    degraded_de = drift_energy[improvement >= 0]

    bins = np.linspace(drift_energy.min(), drift_energy.max(), 15)
    ax.hist(improved_de, bins=bins, alpha=0.6, color='#2196F3', label=f'Ortho improved ({len(improved_de)})')
    ax.hist(degraded_de, bins=bins, alpha=0.6, color='#F44336', label=f'Ortho degraded ({len(degraded_de)})')

    ax.axvline(np.median(improved_de), color='#1565C0', linestyle='--', linewidth=1.5,
               label=f'Improved median={np.median(improved_de):.3f}')
    ax.axvline(np.median(degraded_de), color='#C62828', linestyle='--', linewidth=1.5,
               label=f'Degraded median={np.median(degraded_de):.3f}')

    ax.set_xlabel("Drift Energy", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Drift Energy Distribution: Ortho Improved vs Degraded", fontsize=13)
    ax.legend(fontsize=10)

    # Mann-Whitney U test
    u_stat, u_p = stats.mannwhitneyu(improved_de, degraded_de, alternative='less')
    ax.text(0.98, 0.95, f"Mann-Whitney U p={u_p:.4f}",
            transform=ax.transAxes, fontsize=10, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "drift_distribution_improved_degraded.png"),
                dpi=200, bbox_inches='tight')
    fig.savefig(os.path.join(OUTPUT_DIR, "drift_distribution_improved_degraded.pdf"),
                bbox_inches='tight')
    print("Saved drift_distribution_improved_degraded.png/pdf")
    plt.close(fig)

    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"Summary Statistics")
    print(f"{'='*60}")
    print(f"Ortho improvement: mean={improvement.mean():.1f}%, median={np.median(improvement):.1f}%")
    print(f"  Improved: {n_improved}/{len(improvement)} ({n_improved/len(improvement)*100:.0f}%)")
    print(f"  Degraded: {n_degraded}/{len(improvement)} ({n_degraded/len(improvement)*100:.0f}%)")
    print(f"\nCorrelation (drift_energy vs ortho_improvement):")
    print(f"  r={r_value:.3f}, p={p_value:.3f}")
    print(f"\nDrift energy — improved scenes: {np.median(improved_de):.3f} (median)")
    print(f"Drift energy — degraded scenes: {np.median(degraded_de):.3f} (median)")


if __name__ == "__main__":
    main()
