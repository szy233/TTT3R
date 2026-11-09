"""E7 tables from the logs. Runs no model, and prints whatever has finished so far.
Rulings are written by hand in docs/PREREG_E7.md.

Usage, from the repository root:
    bash rebuttal/iclr2027/scripts/e7_kitti_metrics.sh
    python3 rebuttal/iclr2027/scripts/e7_rulings.py > rebuttal/iclr2027/results/e7.txt
"""
import glob
import hashlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e6_rulings import (  # noqa: E402
    ARKIT, EXPECTED, KITTI, RESULTS, SCANNET, TUM, arm, check_count, finished, line, load_rpe, rpe,
)
from kitti_e5b import load_official  # noqa: E402
from stats_rule import gap_shrink, holm, load_ate  # noqa: E402

SCALES = {"010": 0.10, "020": 0.20, "033": 0.33}
GAINS = f"{RESULTS}/gains"


def registration():
    with open("rebuttal/iclr2027/docs/PREREG_E7.md", "rb") as handle:
        # The registered text ends at the first line holding only "---". Dated results follow it.
        body = handle.read().split(b"\n---\n")[0] + b"\n---\n"
    with open(f"{RESULTS}/e7_registration.sha256") as handle:
        recorded = handle.read().split()
    print(f"registered {recorded[0]} on {' '.join(recorded[2:])}")
    print(f"file now   {hashlib.sha256(body).hexdigest()}")


def check_gain(dataset, name, scale):
    """The logged gain of a fixed-scale arm must equal its scale on every sequence."""
    files = glob.glob(f"{GAINS}/{dataset}/{name}/*.npy")
    if not files:
        print(f"  check gain {dataset}/{name}: no log")
        return
    means = [float(np.load(f).mean()) for f in files]
    worst = max(abs(m - scale) / scale for m in means)
    print(f"  check gain {dataset}/{name} n={len(means)} target={scale} largest gap={100 * worst:.4f}% {'pass' if worst <= 0.005 else 'MISS'}")


def compare_at_each_scale(title, dataset, load, scales=SCALES):
    print(f"\n== {title} ==")
    results, names = [], []
    for tag, scale in scales.items():
        direct, flat = f"dir{tag}", f"const{tag}"
        if not finished(arm(dataset, direct)) or not finished(arm(dataset, flat)):
            print(f"waiting for {dataset}: {direct} or {flat}")
            continue
        check_count(dataset, direct)
        check_gain(dataset, direct, scale)
        results.append(line(f"{direct} vs {flat}", load(direct), load(flat)))
        names.append(tag)
    if results:
        print("Holm over the tests above: " + " ".join(f"{n}={p:.4f}" for n, p in zip(names, holm([r["p"] for r in results]))))
    return dict(zip(names, results))


def rpe_block(dataset, scales=SCALES):
    for tag in scales:
        direct, flat = f"dir{tag}", f"const{tag}"
        if finished(arm(dataset, direct)) and finished(arm(dataset, flat)):
            t_d, r_d = load_rpe(arm(dataset, direct))
            t_f, r_f = load_rpe(arm(dataset, flat))
            line(f"RPE trans {direct} vs {flat}", t_d, t_f)
            line(f"RPE rot   {direct} vs {flat}", r_d, r_f)
            rpe(direct, arm(dataset, direct))
            rpe(flat, arm(dataset, flat))


def against_default(dataset, load, reference, tags):
    print("Against the default ortho, no prediction:")
    for tag in tags:
        if finished(arm(dataset, f"dir{tag}")):
            line(f"dir{tag} vs ortho", load(f"dir{tag}"), reference)


def noise(rotation):
    """Amendment 1. One draw each, so these give a lower bound on the rounding noise."""
    print("\n== Rounding perturbation, Amendment 1 ==")
    pairs = (("orthop", "ortho_check"), ("const033p", "const033"))
    tum = {n: load_ate(arm(TUM, n)) for n in ("orthop", "const033p", "ortho_check", "const033", "const010", "normreplay") if finished(arm(TUM, n))}
    ate = {n: load_ate(arm(KITTI, n)) for n in ("orthop", "const033p", "ortho_check", "const033") if finished(arm(KITTI, n))}

    def drift(label, table, a, b):
        if a in table and b in table:
            line(f"{label} {a} vs {b}", table[a], table[b])
            worst = max(abs(table[a][k] - table[b][k]) / table[b][k] for k in table[b] if k in table[a])
            print(f"  largest single-sequence change {100 * worst:.1f}%")

    for a, b in pairs:
        drift("TUM ATE", tum, a, b)
        drift("KITTI r_rel", rotation, a, b)
        drift("KITTI ATE", ate, a, b)

    def keys(label, table, others):
        for original, swapped in (("ortho_check", "orthop"),):
            for other in others:
                if original in table and other in table:
                    line(f"{label} {original} vs {other}", table[original], table[other])
                if swapped in table and other in table:
                    line(f"{label} {swapped} vs {other}", table[swapped], table[other])
        if "orthop" in table and "const033p" in table:
            line(f"{label} orthop vs const033p", table["orthop"], table["const033p"])

    print("Key comparisons, original then perturbed:")
    keys("TUM ATE", tum, ("const033", "const010", "normreplay"))
    keys("KITTI r_rel", rotation, ("const033", "const010", "normreplay"))


def main():
    registration()

    tum = lambda name: load_ate(arm(TUM, name))
    compare_at_each_scale("P7a, TUM ATE RMSE", TUM, tum)
    against_default(TUM, tum, tum("ortho_check"), SCALES)
    rpe_block(TUM)

    t_new, r_new = load_official(f"{RESULTS}/e6_kitti_metrics.txt")
    t_e7, r_e7 = ({}, {})
    if os.path.exists(f"{RESULTS}/e7_kitti_metrics.txt"):
        t_e7, r_e7 = load_official(f"{RESULTS}/e7_kitti_metrics.txt")
    rotation = {**r_new, **r_e7}
    translation = {**t_new, **t_e7}
    print("\n== P7b, KITTI r_rel ==")
    results, names = [], []
    for tag, scale in SCALES.items():
        direct, flat = f"dir{tag}", f"const{tag}"
        if direct not in rotation:
            print(f"waiting for kitti metrics: {direct}")
            continue
        check_count(KITTI, direct)
        check_gain(KITTI, direct, scale)
        results.append(line(f"r_rel {direct} vs {flat}", rotation[direct], rotation[flat]))
        names.append(tag)
    if results:
        print("Holm over the tests above: " + " ".join(f"{n}={p:.4f}" for n, p in zip(names, holm([r["p"] for r in results]))))
    print("Beside it, no prediction:")
    for tag in names:
        direct, flat = f"dir{tag}", f"const{tag}"
        line(f"r_rel {direct} vs ortho", rotation[direct], rotation["ortho_check"])
        line(f"t_rel {direct} vs {flat}", translation[direct], translation[flat])
        line(f"ATE RMSE {direct} vs {flat}", load_ate(arm(KITTI, direct)), load_ate(arm(KITTI, flat)))
    rpe_block(KITTI, {t: SCALES[t] for t in names})

    two = {"010": 0.10, "033": 0.33}
    scan = lambda name: load_ate(arm(SCANNET, name))
    found = compare_at_each_scale("P7c, ScanNet ATE RMSE", SCANNET, scan, two)
    if len(found) == 2:
        mean, (lo, hi) = gap_shrink(scan("dir033"), scan("const033"), scan("dir010"), scan("const010"))
        print(f"harm at 0.33 minus harm at 0.10: mean={mean:+.4f} CI[{lo:+.4f},{hi:+.4f}] -> {'smaller at 0.10' if lo > 0 else 'not shown smaller'}")
    against_default(SCANNET, scan, scan("ortho_check") if finished(arm(SCANNET, "ortho_check")) else {}, two)
    rpe_block(SCANNET, two)

    ark = lambda name: load_ate(arm(ARKIT, name))
    compare_at_each_scale("ARKitScenes ATE RMSE, no prediction", ARKIT, ark)
    against_default(ARKIT, ark, ark("ortho_check"), SCALES)
    rpe_block(ARKIT)

    print("\n== Flat multipliers against the default ortho on all four datasets ==")
    print("A flat value counts as 'not worse' when the common rule does not call ortho better.")
    for tag in ("010", "033"):
        flat = f"const{tag}"
        rows = []
        if finished(arm(TUM, flat)):
            rows.append(("TUM ATE", tum("ortho_check"), tum(flat)))
        if flat in rotation:
            rows.append(("KITTI r_rel", rotation["ortho_check"], rotation[flat]))
        if finished(arm(SCANNET, flat)):
            rows.append(("ScanNet ATE", scan("ortho_check"), scan(flat)))
        if finished(arm(ARKIT, flat)):
            rows.append(("ARKitScenes ATE", ark("ortho_check"), ark(flat)))
        verdicts = [line(f"{label}: ortho vs {flat}", a, b)["verdict"] for label, a, b in rows]
        complete = len(rows) == 4
        print(f"  {flat}: {'not worse than ortho on all four' if complete and 'a' not in verdicts else 'worse on at least one' if 'a' in verdicts else 'incomplete'}")
    if finished(arm(ARKIT, "const020")):
        line("ARKitScenes ATE: ortho vs const020", ark("ortho_check"), ark("const020"))
    print(f"expected sequence counts: {EXPECTED}")
    noise(rotation)


if __name__ == "__main__":
    main()
