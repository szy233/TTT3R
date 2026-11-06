#!/usr/bin/env python3
"""Extract TUM per-sequence ATE RMSE and split sitting (mild dynamics)
vs walking (severe dynamics). Zero new GPU runs -- reads existing eval output."""
import os, re, sys

ROOT = "/home/szy/research/TTT3R/eval_results/relpose"
METHODS = ["cut3r", "ttt3r", "ddd3r_constant", "ddd3r_brake", "ddd3r"]
SEQS = ["sitting_halfsphere", "sitting_rpy", "sitting_static", "sitting_xyz",
        "walking_halfsphere", "walking_rpy", "walking_static", "walking_xyz"]


def read_rmse(path):
    """First 'rmse' line belongs to the APE (ATE) block."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        txt = f.read()
    # APE block is first; grab its rmse
    m = re.search(r"APE w\.r\.t\..*?\n\s*rmse\s+([\d.eE+-]+)", txt, re.S)
    if m:
        return float(m.group(1))
    return None


for dataset in ["tum_s1_1000", "tum_s1_90"]:
    base = os.path.join(ROOT, dataset)
    if not os.path.isdir(base):
        print(f"\n### {dataset}: NOT FOUND\n")
        continue
    print(f"\n=== {dataset} (ATE RMSE, m) ===")
    header = f"{'seq':<22}" + "".join(f"{m:>17}" for m in METHODS)
    print(header)
    table = {}
    for seq in SEQS:
        row = []
        for meth in METHODS:
            p = os.path.join(base, meth,
                             f"rgbd_dataset_freiburg3_{seq}_eval_metric.txt")
            row.append(read_rmse(p))
        table[seq] = row
        cells = "".join(f"{(f'{v:.4f}' if v is not None else 'NA'):>17}" for v in row)
        print(f"{seq:<22}{cells}")

    def group_mean(prefix):
        out = []
        for i in range(len(METHODS)):
            vals = [table[s][i] for s in SEQS
                    if s.startswith(prefix) and table[s][i] is not None]
            out.append(sum(vals) / len(vals) if vals else None)
        return out

    print("-" * len(header))
    for label, prefix in [("MEAN sitting (mild)", "sitting"),
                          ("MEAN walking (severe)", "walking"),
                          ("MEAN all 8", "")]:
        g = group_mean(prefix)
        cells = "".join(f"{(f'{v:.4f}' if v is not None else 'NA'):>17}" for v in g)
        print(f"{label:<22}{cells}")

    # relative improvement vs cut3r and vs ttt3r on each group
    print()
    for label, prefix in [("sitting (mild)", "sitting"),
                          ("walking (severe)", "walking"),
                          ("all 8", "")]:
        g = group_mean(prefix)
        if g[0] is None:
            continue
        parts = []
        for i, m in enumerate(METHODS):
            if g[i] is None:
                continue
            vs_cut = (g[i] - g[0]) / g[0] * 100
            vs_ttt = (g[i] - g[1]) / g[1] * 100 if g[1] else float("nan")
            parts.append(f"{m}={g[i]:.4f}(vsCUT3R {vs_cut:+.1f}%, vsTTT3R {vs_ttt:+.1f}%)")
        print(f"[{label}] " + "; ".join(parts))
