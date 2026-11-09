"""Paired tests behind two MDAW sentences that had only means in a result file. Not registered.

Reads the same relpose logs as paired_stats.py. No model is run.

Usage, from the repository root:
    python3 rebuttal/iclr2027/scripts/reply_sources.py > rebuttal/iclr2027/results/reply_sources.txt
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats_rule import bootstrap, wilcoxon  # noqa: E402

ROOT = "/home/szy/research/TTT3R/eval_results/relpose"
SEQ = re.compile(r"^(\S+?)\s*\|\s*ATE:\s*([0-9.]+)")
GRID = [
    "ttt3r_ortho_an0.3_ad005",
    "ttt3r_ortho_an05_ad00",
    "ttt3r_ortho_an05_ad005",
    "ttt3r_ortho_an05_ad005_b0.9",
    "ttt3r_ortho_an05_ad005_b0.99",
    "ttt3r_ortho_an05_ad0.1",
    "ttt3r_ortho_an05_ad0.2",
    "ttt3r_ortho_an0.7_ad005",
]


def load(ds, cfg):
    out = {}
    for line in open(f"{ROOT}/{ds}/{cfg}/_error_log.txt", errors="ignore"):
        m = SEQ.match(line)
        if m:
            out[m.group(1).split("-", 1)[-1]] = float(m.group(2))
    return out


def compare(ds, a, b):
    A, B = load(ds, a), load(ds, b)
    keys = sorted(set(A) & set(B))
    d = [A[k] - B[k] for k in keys]
    ma = sum(A[k] for k in keys) / len(keys)
    mb = sum(B[k] for k in keys) / len(keys)
    lo, hi = bootstrap(d)
    print(
        f"{ds:16s} {a:30s} vs {b:16s} n={len(keys):3d} {ma:.4f} vs {mb:.4f} "
        f"({100 * (ma / mb - 1):+6.1f}%) wins={sum(x < 0 for x in d)}/{len(keys)} "
        f"CI[{lo:+.4f},{hi:+.4f}] p={wilcoxon(d):.4f}"
    )


print("== TUM 1000f grid configs against brake tau=1 ==")
for cfg in GRID:
    compare("tum_s1_1000", cfg, "ddd3r_brake")

print("\n== Brake tau=2 against const 0.5 ==")
for ds in ("tum_s1_1000", "scannet_s3_1000"):
    compare(ds, "ttt3r_momentum_inv_t2", "ddd3r_constant")
