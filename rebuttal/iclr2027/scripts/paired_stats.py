"""Paired scene-level statistics for the ICLR 2027 rebuttal, from existing relpose logs only.

No model is run. Reads per-sequence ATE from each config's _error_log.txt, then prints
paired comparisons (exact Wilcoxon for n <= 16, normal approximation above that, paired
bootstrap 95% CI of the mean difference), a nested leave-one-out selection over the TUM
hyperparameter grid, and the RPE averages the paper does not report.

Usage: python3 paired_stats.py [eval_results/relpose root] [ttsa results/relpose root]
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats_rule import bootstrap, wilcoxon  # noqa: E402

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/home/szy/research/TTT3R/eval_results/relpose"
TTSA = sys.argv[2] if len(sys.argv) > 2 else (
    "/home/szy/research/TTT3R/.claude/worktrees/understand-full-project-14f204/rebuttal/results/relpose"
)
SEQ = re.compile(r"^(\S+?)\s*\|\s*ATE:\s*([0-9.]+)")
AVG = re.compile(r"Average ATE:\s*([0-9.]+),\s*Average RPE trans:\s*([0-9.]+),\s*Average RPE rot:\s*([0-9.]+)")


def load(base, ds, cfg):
    path = f"{base}/{ds}/{cfg}/_error_log.txt"
    if not os.path.exists(path):
        return None
    out = {}
    for line in open(path, errors="ignore"):
        m = SEQ.match(line)
        if m:
            out[m.group(1).split("-", 1)[-1]] = float(m.group(2))
    return out


def average(base, ds, cfg):
    path = f"{base}/{ds}/{cfg}/_error_log.txt"
    if not os.path.exists(path):
        return None
    last = None
    for line in open(path, errors="ignore"):
        m = AVG.search(line)
        if m:
            last = tuple(float(x) for x in m.groups())
    return last


def compare(ds, a, b, base_a=ROOT, base_b=ROOT):
    A, B = load(base_a, ds, a), load(base_b, ds, b)
    if not A or not B:
        print(f"{ds:16s} {a} vs {b}: missing")
        return
    keys = sorted(set(A) & set(B))
    d = [A[k] - B[k] for k in keys]
    ma = sum(A[k] for k in keys) / len(keys)
    mb = sum(B[k] for k in keys) / len(keys)
    lo, hi = bootstrap(d)
    print(
        f"{ds:16s} {a:20s} vs {b:20s} n={len(keys):3d} {ma:.4f} vs {mb:.4f} "
        f"({100 * (ma / mb - 1):+6.1f}%) wins={sum(x < 0 for x in d)}/{len(keys)} "
        f"CI[{lo:+.4f},{hi:+.4f}] p={wilcoxon(d):.4f}"
    )


print("== Paired ATE ==")
for a, b in [
    ("ddd3r", "ddd3r_brake"),
    ("ddd3r", "ttt3r_random_p033"),
    ("ddd3r", "ddd3r_constant"),
    ("ddd3r_brake", "ttt3r_random_p033"),
    ("ddd3r_brake", "ddd3r_constant"),
]:
    compare("tum_s1_1000", a, b)
for a, b in [
    ("ddd3r", "ddd3r_brake"),
    ("ddd3r", "ddd3r_constant"),
    ("ddd3r", "ttt3r"),
    ("ddd3r_brake", "ddd3r_constant"),
]:
    compare("scannet_s3_1000", a, b)

print("\n== Per-sequence TUM 1000f ATE ==")
ortho = load(ROOT, "tum_s1_1000", "ddd3r")
brake = load(ROOT, "tum_s1_1000", "ddd3r_brake")
c33 = load(ROOT, "tum_s1_1000", "ttt3r_random_p033")
for k in sorted(ortho):
    print(f"{k.split('freiburg3_')[-1]:20s} ortho={ortho[k]:.5f} brake={brake[k]:.5f} const0.33={c33[k]:.5f}")

print("\n== Nested leave-one-out selection on the TUM grid ==")
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
runs = {c: load(ROOT, "tum_s1_1000", c) for c in GRID}
runs = {c: v for c, v in runs.items() if v and len(v) == 8}
for c, v in runs.items():
    print(f"  {c:32s} mean={sum(v.values()) / 8:.4f}")
seqs = sorted(next(iter(runs.values())))
picks, held, vs_brake, vs_c33 = [], [], [], []
for s in seqs:
    best = min(runs, key=lambda c: sum(runs[c][x] for x in seqs if x != s))
    picks.append(best)
    held.append(runs[best][s])
    vs_brake.append(runs[best][s] - brake[s])
    vs_c33.append(runs[best][s] - c33[s])
print(f"  picks: {sorted(set(picks))}")
for s, pick, value in zip(seqs, picks, held):
    print(f"  {s.split('freiburg3_')[-1]:20s} pick={pick:32s} held-out={value:.5f}")
print(f"  held-out mean={sum(held) / 8:.4f} brake={sum(brake.values()) / 8:.4f} const0.33={sum(c33.values()) / 8:.4f}")
for name, d in (("brake", vs_brake), ("const0.33", vs_c33)):
    lo, hi = bootstrap(d)
    print(f"  vs {name}: wins={sum(x < 0 for x in d)}/8 CI[{lo:+.4f},{hi:+.4f}] p={wilcoxon(d):.4f}")

print("\n== Same configuration, two runs (noise floor) ==")
for ds, a, b in [
    ("tum_s1_1000", "ddd3r", "ttt3r_ortho_an05_ad005"),
    ("tum_s1_1000", "ddd3r_constant", "ttt3r_random"),
    ("scannet_s3_1000", "ddd3r_constant", "ttt3r_random"),
]:
    x, y = average(ROOT, ds, a), average(ROOT, ds, b)
    print(f"  {ds:16s} {a}={x[0]:.5f} {b}={y[0]:.5f} ({100 * abs(x[0] - y[0]) / min(x[0], y[0]):.1f}%)")

print("\n== Brake temperature ==")
for ds in ("tum_s1_1000", "scannet_s3_1000"):
    for c in ("ddd3r_brake", "ttt3r_momentum_inv_t1", "ttt3r_momentum_inv_t2", "ddd3r_constant"):
        print(f"  {ds:16s} {c:24s} ATE={average(ROOT, ds, c)[0]:.5f}")

print("\n== RPE, not reported in the paper ==")
for ds in ("tum_s1_1000", "scannet_s3_1000"):
    base = average(ROOT, ds, "cut3r")
    for c in ("cut3r", "ttt3r", "ddd3r_constant", "ddd3r_brake", "ddd3r"):
        v = average(ROOT, ds, c)
        print(
            f"  {ds:16s} {c:16s} ATE={v[0]:.4f} RPEt={v[1]:.4f} ({100 * (v[1] / base[1] - 1):+6.1f}%) "
            f"RPEr={v[2]:.4f} ({100 * (v[2] / base[2] - 1):+6.1f}%)"
        )

print("\n== TTSA3R vs TTT3R ==")
for ds in ("tum", "tum_s1_1000", "scannet_s3_90", "scannet_s3_1000", "sintel", "arkit_s1_500"):
    t = load(TTSA, ds, "ttsa3r")
    b = load(ROOT, ds, "ttt3r") or load(TTSA, ds, "ttt3r")
    if t and b:
        keys = sorted(set(t) & set(b))
        d = [t[k] - b[k] for k in keys]
        print(
            f"  {ds:16s} n={len(keys):3d} ttsa={sum(t[k] for k in keys) / len(keys):.5f} "
            f"ttt3r={sum(b[k] for k in keys) / len(keys):.5f} wins={sum(x < 0 for x in d)} p={wilcoxon(d):.4f}"
        )
