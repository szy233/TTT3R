#!/usr/bin/env python3
"""
Generate every quantitative table the rebuttal needs, from existing eval output.
No GPU, no new inference -- pure extraction, so numbers are auditable.

Tables produced:
  T1  TUM 1000f split by dynamics severity (sitting = mild, walking = severe)
      -> answers DpBu R2-C2 "complete absence of dynamic-scene evaluation"
  T2  TTSA3R head-to-head on long sequences
      -> answers 1ake R3-C2 "TAUM has no head-to-head row"

Usage:  python rebuttal/scripts/make_tables.py
"""
import json
import os
import re

MAIN = "/home/szy/research/TTT3R/eval_results"
REB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

TUM_SEQS = ["sitting_halfsphere", "sitting_rpy", "sitting_static", "sitting_xyz",
            "walking_halfsphere", "walking_rpy", "walking_static", "walking_xyz"]

# display name -> (root, dirname)
METHODS = [
    ("CUT3R",        MAIN, "cut3r"),
    ("TTT3R",        MAIN, "ttt3r"),
    ("TTSA3R",       REB,  "ttsa3r"),      # new, this rebuttal
    ("DDD3R_const",  MAIN, "ddd3r_constant"),
    ("DDD3R_brake",  MAIN, "ddd3r_brake"),
    ("DDD3R_ortho",  MAIN, "ddd3r"),
]


def ape_rmse(path):
    """RMSE of the APE (= ATE) block, which is the first block in the file."""
    if not os.path.exists(path):
        return None
    txt = open(path).read()
    m = re.search(r"APE w\.r\.t\..*?\n\s*rmse\s+([\d.eE+-]+)", txt, re.S)
    return float(m.group(1)) if m else None


def tum_path(root, meth, seq):
    return os.path.join(root, "relpose", "tum_s1_1000", meth,
                        f"rgbd_dataset_freiburg3_{seq}_eval_metric.txt")


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def fmt(v, nd=4):
    return f"{v:.{nd}f}" if v is not None else "n/a"


def pct(v, ref):
    if v is None or ref in (None, 0):
        return "n/a"
    return f"{(v - ref) / ref * 100:+.1f}%"


def table1_dynamics():
    print("\n" + "=" * 78)
    print("T1  TUM 1000f -- ATE RMSE (m), split by dynamics severity")
    print("    All 8 TUM sequences are freiburg3 sitting_*/walking_* = TUM RGB-D")
    print("    'Dynamic Objects' category. walking_* = two people walking through view.")
    print("=" * 78)
    rows = {}
    for name, root, d in METHODS:
        vals = {s: ape_rmse(tum_path(root, d, s)) for s in TUM_SEQS}
        rows[name] = vals

    hdr = f"{'sequence':<22}" + "".join(f"{n:>14}" for n, _, _ in METHODS)
    print(hdr)
    for s in TUM_SEQS:
        print(f"{s:<22}" + "".join(f"{fmt(rows[n][s]):>14}" for n, _, _ in METHODS))
    print("-" * len(hdr))

    groups = [("sitting (mild)", "sitting"), ("walking (SEVERE)", "walking"), ("all 8", "")]
    summary = {}
    for label, pre in groups:
        gm = {n: mean([rows[n][s] for s in TUM_SEQS if s.startswith(pre)])
              for n, _, _ in METHODS}
        summary[label] = gm
        print(f"{label:<22}" + "".join(f"{fmt(gm[n]):>14}" for n, _, _ in METHODS))

    print("\nRelative to baselines:")
    for label, _ in groups:
        gm = summary[label]
        cut, ttt = gm.get("CUT3R"), gm.get("TTT3R")
        print(f"  [{label}]")
        for n, _, _ in METHODS:
            if n in ("CUT3R",):
                continue
            print(f"    {n:<14} {fmt(gm[n]):>8}   vs CUT3R {pct(gm[n], cut):>8}"
                  f"   vs TTT3R {pct(gm[n], ttt):>8}")
    return summary


def table2_ttsa3r():
    print("\n" + "=" * 78)
    print("T2  TTSA3R head-to-head (Reviewer 1ake R3-C2)")
    print("    Prediction from M2: TAUM collapses to ~constant dampening, so TTSA3R")
    print("    should land near DDD3R_const, NOT near DDD3R_brake/ortho.")
    print("=" * 78)
    for ds, label in [("tum_s1_1000", "TUM 1000f"), ("scannet_s3_1000", "ScanNet 1000f")]:
        print(f"\n  {label}:")
        for name, root, d in METHODS:
            base = os.path.join(root, "relpose", ds, d)
            if not os.path.isdir(base):
                print(f"    {name:<14} (not run)")
                continue
            files = [f for f in os.listdir(base) if f.endswith("_eval_metric.txt")]
            vals = [ape_rmse(os.path.join(base, f)) for f in files]
            vals = [v for v in vals if v is not None]
            if not vals:
                print(f"    {name:<14} (no metrics yet, {len(files)} files)")
                continue
            print(f"    {name:<14} {sum(vals)/len(vals):.4f}   (n={len(vals)} seqs)")


def sintel_ttsa3r():
    print("\n" + "=" * 78)
    print("T3  Sintel relpose -- TTSA3R (complete)")
    print("=" * 78)
    base = os.path.join(REB, "relpose", "sintel", "ttsa3r")
    if not os.path.isdir(base):
        print("  not run")
        return
    files = sorted(f for f in os.listdir(base) if f.endswith("_eval_metric.txt"))
    vals = []
    for f in files:
        v = ape_rmse(os.path.join(base, f))
        if v is not None:
            vals.append(v)
            print(f"    {f.replace('_eval_metric.txt',''):<16} {v:.4f}")
    if vals:
        print(f"    {'MEAN':<16} {sum(vals)/len(vals):.4f}  (n={len(vals)})")
        print("    paper reference: CUT3R 0.209, TTT3R 0.209, const 0.220, "
              "brake 0.237, ortho 0.236")


if __name__ == "__main__":
    table1_dynamics()
    table2_ttsa3r()
    sintel_ttsa3r()
    print()
