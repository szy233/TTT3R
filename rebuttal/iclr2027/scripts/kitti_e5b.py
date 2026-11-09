"""E5b tables from the KITTI rerun. Reads logs only and runs no model.

Usage, from the repository root:
    python3 rebuttal/iclr2027/scripts/kitti_e5b.py > rebuttal/iclr2027/results/kitti_e5b.txt
"""
import itertools
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from stats_rule import compare, load_ate, load_ate_mean, load_ate_rmse  # noqa: E402

RESULTS = "rebuttal/iclr2027/results"
ROOT = f"{RESULTS}/relpose/kitti_odo_full"
ARMS = ["cut3r", "ttt3r_plain", "const05", "brake_t1", "ortho_plain"]
TOLERANCE = 0.005
LOGGED = "/home/szy/research/TTT3R/eval_results/relpose"
TTSA3R = (
    "/home/szy/research/TTT3R/.claude/worktrees/understand-full-project-14f204"
    "/rebuttal/results/relpose/kitti_odom/ttsa3r"
)
GAMMAS = ["0.5", "1", "2", "3"]
AVERAGE = re.compile(r"Average ATE:\s*([0-9.]+),\s*Average RPE trans:\s*([0-9.]+),\s*Average RPE rot:\s*([0-9.]+)")

# Amendment 2, values copied as registered. const05's references came from the logged
# "constant" folder, whose alpha the diagnostic below tests.
REGISTERED = {
    "cut3r": {"04": 21.84807, "03": 150.29048},
    "ttt3r_plain": {"04": 11.06465, "03": 103.25776},
    "const05": {"04": 5.09581, "03": 60.68427},
    "brake_t1": {"04": 5.55621},
    "ortho_plain": {"04": 17.78537, "03": 39.90691},
}
OFFICIAL = re.compile(r"^\s*(\S+)\s*\|\s*(\d\d)\s*\|\s*t_rel:\s*([0-9.]+)%\s*\|\s*r_rel:\s*([0-9.]+)")


def load_official(path):
    t_rel, r_rel = {}, {}
    with open(path, errors="ignore") as handle:
        for line in handle:
            match = OFFICIAL.match(line)
            if match:
                arm, seq = match.group(1), match.group(2)
                t_rel.setdefault(arm, {})[seq] = float(match.group(3))
                r_rel.setdefault(arm, {})[seq] = float(match.group(4))
    return t_rel, r_rel


def mean(values):
    values = list(values)
    return sum(values) / len(values)


def per_sequence(title, table, digits=2):
    print(f"\n== {title} ==")
    print(f"{'seq':4s}" + "".join(f"{arm:>13s}" for arm in ARMS))
    for seq in sorted(table[ARMS[0]]):
        print(f"{seq:4s}" + "".join(f"{table[arm][seq]:13.{digits}f}" for arm in ARMS))
    base = mean(table["cut3r"].values())
    print(f"{'avg':4s}" + "".join(f"{mean(table[arm].values()):13.{digits}f}" for arm in ARMS))
    print(f"{'vs0':4s}" + "".join(f"{100 * (mean(table[arm].values()) / base - 1):+12.1f}%" for arm in ARMS))
    order = sorted(ARMS, key=lambda arm: mean(table[arm].values()))
    print("rank: " + " < ".join(order))


def paired(title, table, pairs):
    print(f"\n== {title} ==")
    for a, b in pairs:
        r = compare(table[a], table[b])
        print(
            f"{a:12s} vs {b:12s} n={r['n']:2d} {r['mean_a']:9.3f} vs {r['mean_b']:9.3f} "
            f"({100 * r['gap']:+6.1f}%) wins={r['wins_a']:2d}/{r['n']} "
            f"CI[{r['ci'][0]:+9.3f},{r['ci'][1]:+9.3f}] p={r['p']:.4f} -> {r['verdict']}"
        )


def main():
    ate = {arm: load_ate(f"{ROOT}/{arm}") for arm in ARMS}
    ate_mean = {arm: load_ate_mean(f"{ROOT}/{arm}") for arm in ARMS}
    t_rel, r_rel = load_official(f"{RESULTS}/kitti_official_metrics.txt")

    print("== Protocol check, registered references ==")
    misses = []
    for arm, references in REGISTERED.items():
        for seq, reference in references.items():
            value = ate[arm][seq]
            gap = abs(value - reference) / reference
            verdict = "ok" if gap <= TOLERANCE else "MISS"
            if verdict == "MISS":
                misses.append(f"{arm}/{seq}")
            print(f"{arm:12s} seq{seq} registered={reference:10.5f} rerun={value:10.5f} gap={100 * gap:7.3f}% {verdict}")
    print(f"misses: {misses or 'none'}")

    diagnostic = f"{RESULTS}/relpose/kitti_odo_full__constcheck/const033"
    if os.path.exists(f"{diagnostic}/_error_log.txt"):
        print("\n== Diagnostic, alpha=0.33 against the same references ==")
        found = load_ate(diagnostic)
        for seq, value in sorted(found.items()):
            reference = REGISTERED["const05"][seq]
            print(f"const033 seq{seq} reference={reference:10.5f} run={value:10.5f} gap={100 * abs(value - reference) / reference:7.3f}%")

    print("\n== TTSA3R, sequence 04 rerun against the earlier run ==")
    earlier = load_ate_rmse(TTSA3R)
    rerun = load_ate(f"{RESULTS}/relpose/kitti_odo_full__ttsacheck04/ttsa3r_log")["04"]
    print(f"earlier={earlier['04']:.6f} rerun={rerun:.5f} gap={100 * abs(rerun - earlier['04']) / earlier['04']:.4f}%")
    print(f"11-sequence ATE RMSE {mean(earlier.values()):.3f}, ATE mean {mean(load_ate_mean(TTSA3R).values()):.3f}")
    print("per sequence: " + " ".join(f"{k}={v:.2f}" for k, v in sorted(earlier.items())))
    for arm in ARMS:
        r = compare(earlier, ate[arm])
        print(
            f"ttsa3r vs {arm:12s} n={r['n']:2d} {r['mean_a']:8.2f} vs {r['mean_b']:8.2f} ({100 * r['gap']:+6.1f}%) "
            f"wins={r['wins_a']:2d} CI[{r['ci'][0]:+8.2f},{r['ci'][1]:+8.2f}] p={r['p']:.4f} -> {r['verdict']}"
        )

    print("\n== Table 5 gamma rows, rerun against the logged runs ==")
    for gamma in GAMMAS:
        found = load_ate(f"{RESULTS}/relpose/kitti_odo_full__gammacheck/ortho_g{gamma}")
        logged = {
            "04": load_ate(f"{LOGGED}/kitti_odom/steep_v2_g{gamma}")["04"],
            "03": load_ate(f"{LOGGED}/kitti_odom_03/steep_v2_g{gamma}")["03"],
        }
        print(f"gamma={gamma:3s} " + " ".join(f"seq{s}: logged={logged[s]:.5f} rerun={found[s]:.5f}" for s in ("04", "03")))

    print("\n== RPE averages over the 11 sequences ==")
    base = None
    for arm in ARMS:
        with open(f"{ROOT}/{arm}/_error_log.txt", errors="ignore") as handle:
            values = [AVERAGE.search(line) for line in handle]
        last = [m for m in values if m][-1]
        trans, rot = float(last.group(2)), float(last.group(3))
        base = base or (trans, rot)
        print(f"{arm:12s} RPE trans={trans:.3f} ({trans / base[0]:.2f}x) RPE rot={rot:.3f} ({rot / base[1]:.2f}x)")

    per_sequence("ATE RMSE, the registered metric", ate)
    per_sequence("ATE mean, the metric the report's aggregate printed", ate_mean)
    per_sequence("Official t_rel (%), mean over sequences", t_rel)
    per_sequence("Official r_rel (deg/100m), mean over sequences", r_rel, digits=3)

    all_pairs = list(itertools.combinations(ARMS, 2))
    paired("Paired ATE RMSE, common rule", ate, all_pairs)
    paired("Paired ATE mean, common rule", ate_mean, all_pairs)
    paired("Paired t_rel, common rule", t_rel, all_pairs)
    paired("Paired r_rel, common rule", r_rel, all_pairs)


if __name__ == "__main__":
    main()
