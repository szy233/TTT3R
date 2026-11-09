"""E1-E4 tables from the logs. Runs no model. Rulings are written by hand in PREREG_E1_E5.md.

Usage, from the repository root:
    python3 rebuttal/iclr2027/scripts/e1_e4_rulings.py > rebuttal/iclr2027/results/e1_e4.txt
"""
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats_rule import compare, frozen_verdict, gate_variance_parts, load_ate  # noqa: E402

RESULTS = "rebuttal/iclr2027/results"
RELPOSE = f"{RESULTS}/relpose"
LOGGED = "/home/szy/research/TTT3R/eval_results/relpose"
LOGGED_ARKIT = (
    "/home/szy/research/TTT3R/.claude/worktrees/understand-full-project-14f204"
    "/rebuttal/results/relpose/arkit_s1_500"
)
TUM, SCANNET, ARKIT = "tum_s1_1000", "scannet_s3_1000", "arkit_s1_500"
FIRST20 = "scannet_s3_1000__first20"
AVERAGE = re.compile(r"Average ATE:\s*([0-9.]+),\s*Average RPE trans:\s*([0-9.]+),\s*Average RPE rot:\s*([0-9.]+)")
INSTRUMENT_REFERENCE = 0.05454
INSTRUMENT_TOLERANCE = 0.022


def averages(folder):
    """ATE, RPE trans and RPE rot from the last Average line of a run."""
    with open(os.path.join(folder, "_error_log.txt"), errors="ignore") as handle:
        found = [AVERAGE.search(line) for line in handle]
    return tuple(float(x) for x in [m for m in found if m][-1].groups())


def arm(dataset, name):
    return f"{RELPOSE}/{dataset}/{name}"


def logged(dataset, name):
    return f"{LOGGED_ARKIT}/{name}" if dataset == ARKIT else f"{LOGGED}/{dataset}/{name}"


def line(label, a, b):
    r = compare(a, b)
    print(
        f"{label:44s} n={r['n']:2d} {r['mean_a']:.5f} vs {r['mean_b']:.5f} ({100 * r['gap']:+6.1f}%) "
        f"wins={r['wins_a']:2d}/{r['n']} CI[{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}] p={r['p']:.4f} -> {r['verdict']}"
    )
    return r


def rpe_row(label, folder):
    ate, trans, rot = averages(folder)
    print(f"  {label:24s} ATE={ate:.5f} RPE trans={trans:.5f} RPE rot={rot:.5f}")


def mean_gain(dataset, name, keep=None):
    """Mean gain over each sequence's frames, then over sequences."""
    values = {}
    for path in glob.glob(f"{RESULTS}/gains/{dataset}/{name}/*.npy"):
        seq = os.path.splitext(os.path.basename(path))[0]
        if keep is None or seq in keep:
            values[seq] = float(np.load(path).mean())
    return values


def instrument():
    print("== Instrument and reference check ==")
    ortho = averages(arm(TUM, "ortho_check"))[0]
    gap =abs(ortho - INSTRUMENT_REFERENCE) / INSTRUMENT_REFERENCE
    print(f"TUM ortho_check {ortho:.5f} against registered {INSTRUMENT_REFERENCE} gap={100 * gap:.3f}% "
          f"{'pass' if gap <= INSTRUMENT_TOLERANCE else 'FAIL'}")
    for dataset in (TUM, SCANNET, ARKIT):
        fresh = load_ate(arm(dataset, "ortho_check"))
        old = load_ate(logged(dataset, "ddd3r"))
        keys = sorted(set(fresh) & set(old))
        worst = max(abs(fresh[k] - old[k]) / old[k] for k in keys)
        print(f"{dataset:16s} fresh ortho vs logged ddd3r n={len(keys)} largest per-sequence gap={100 * worst:.4f}%")


def e1():
    print("\n== E1b, normreplay (the verdict on direction). a=ortho, b=normreplay ==")
    for dataset in (TUM, SCANNET, ARKIT):
        line(f"{dataset} ortho vs normreplay", load_ate(arm(dataset, "ortho_check")), load_ate(arm(dataset, "normreplay")))

    print("\n== E1, normmatch (closed-loop scalar rule). a=ortho, b=normmatch ==")
    for dataset in (TUM, SCANNET, ARKIT):
        if os.path.exists(f"{arm(dataset, 'normmatch')}/_error_log.txt"):
            line(f"{dataset} ortho vs normmatch", load_ate(arm(dataset, "ortho_check")), load_ate(arm(dataset, "normmatch")))

    print("\n== Manipulation check, mean gain per run (registered: report a gap above 10%) ==")
    for dataset in (TUM, SCANNET, ARKIT):
        ortho, match = mean_gain(dataset, "ortho_check"), mean_gain(dataset, "normmatch")
        keys = sorted(set(ortho) & set(match))
        if keys:
            a, b = (sum(g[k] for k in keys) / len(keys) for g in (ortho, match))
            print(f"{dataset:16s} n={len(keys):2d} ortho gain={a:.4f} normmatch gain={b:.4f} gap={100 * abs(a - b) / a:.1f}%")

    print("\n== normreplay and normmatch against the scalar baselines, not registered, no prediction ==")
    for dataset in (TUM, SCANNET, ARKIT):
        replay = load_ate(arm(dataset, "normreplay"))
        for name in ("ddd3r_brake", "ddd3r_constant", "ttt3r"):
            line(f"{dataset} normreplay vs {name}", replay, load_ate(logged(dataset, name)))

    print("\n== RPE, E1 arms ==")
    for dataset in (TUM, SCANNET, ARKIT):
        print(dataset)
        for name in ("ortho_check", "normreplay", "normmatch"):
            if os.path.exists(f"{arm(dataset, name)}/_error_log.txt"):
                rpe_row(name, arm(dataset, name))
        for name in ("ttt3r", "ddd3r_constant", "ddd3r_brake"):
            rpe_row(name + " (logged)", logged(dataset, name))


def e2():
    print("\n== E2, projection timing, no prediction. a=ortho (updated d_t), b=prevdir (d_t-1) ==")
    for dataset in (TUM, SCANNET):
        line(f"{dataset} ortho vs prevdir", load_ate(arm(dataset, "ortho_check")), load_ate(arm(dataset, "prevdir")))
        rpe_row("ortho", arm(dataset, "ortho_check"))
        rpe_row("prevdir", arm(dataset, "prevdir"))


def e3():
    print("\n== E3a, gate variance, averaged over sequences ==")
    for dataset in (TUM, FIRST20):
        for method in ("ttt3r", "ttsa3r"):
            parts = [gate_variance_parts(np.load(p)) for p in sorted(glob.glob(f"{RESULTS}/gates/{dataset}/{method}/*.npy"))]
            avg = {k: sum(p[k] for p in parts) / len(parts) for k in parts[0]}
            print(
                f"{dataset:26s} {method:7s} n={len(parts):2d} total={avg['total']:.6f} "
                f"token={100 * avg['token'] / avg['total']:5.1f}% frame={100 * avg['frame'] / avg['total']:5.1f}% "
                f"residual={100 * avg['residual'] / avg['total']:5.1f}% "
                f"token_temporal_std={avg['token_temporal_std']:.4f} frame_mean_std={avg['frame_mean_std']:.4f}"
            )

    print("\n== E3b, frozen against live. a=frozen, b=live ==")
    for dataset in (TUM, FIRST20):
        for method in ("ttt3r", "ttsa3r"):
            r = line(
                f"{dataset} {method} frozen vs live",
                load_ate(arm(dataset, f"{method}_frozen")),
                load_ate(arm(dataset, f"{method}_log")),
            )
            margin = 0.05 * r["mean_b"]
            verdict = frozen_verdict(r["mean_b"], r["mean_a"], r["ci"][1], r["p"])
            print(f"    upper CI {r['ci'][1]:+.4f} against 5% of live {margin:.4f} -> {verdict}")


def e4():
    print("\n== E4, ScanNet. a=const033, b=brake tau=1 (logged) ==")
    const033 = arm(SCANNET, "const033")
    brake = logged(SCANNET, "ddd3r_brake")
    line("scannet const033 vs brake", load_ate(const033), load_ate(brake))
    line("scannet const033 vs const05", load_ate(const033), load_ate(logged(SCANNET, "ddd3r_constant")))
    rpe_row("const033", const033)
    rpe_row("brake (logged)", brake)


def reviewer_pairs():
    """Comparisons the reviews ask about directly, from runs that already exist."""
    print("\n== Ortho against the methods named in the reviews, not registered ==")
    ttsa_scannet = (
        "/home/szy/research/TTT3R/.claude/worktrees/understand-full-project-14f204"
        "/rebuttal/results/relpose/scannet_s3_1000/ttsa3r"
    )
    scannet = load_ate(arm(SCANNET, "ortho_check"))
    arkit = load_ate(arm(ARKIT, "ortho_check"))
    line("scannet ortho vs ttt3r", scannet, load_ate(logged(SCANNET, "ttt3r")))
    line("scannet ortho vs ttsa3r", scannet, load_ate(ttsa_scannet))
    line("scannet ortho vs brake", scannet, load_ate(logged(SCANNET, "ddd3r_brake")))
    line("scannet ortho vs const05", scannet, load_ate(logged(SCANNET, "ddd3r_constant")))
    line("arkit ortho vs ttt3r", arkit, load_ate(logged(ARKIT, "ttt3r")))
    line("arkit ortho vs ttsa3r", arkit, load_ate(logged(ARKIT, "ttsa3r")))
    line("arkit ortho vs brake", arkit, load_ate(logged(ARKIT, "ddd3r_brake")))
    line("arkit ortho vs const05", arkit, load_ate(logged(ARKIT, "ddd3r_constant")))
    print("\n== Brake against TTT3R, for the closing claim, not registered ==")
    for dataset in (TUM, SCANNET, ARKIT):
        line(f"{dataset} brake vs ttt3r", load_ate(logged(dataset, "ddd3r_brake")), load_ate(logged(dataset, "ttt3r")))
        rpe_row("brake (logged)", logged(dataset, "ddd3r_brake"))
        rpe_row("ttt3r (logged)", logged(dataset, "ttt3r"))


def online_selectors():
    """How many logged runs are lower than brake on each dataset. Brake is a selected winner."""
    print("\n== Logged runs against brake on TUM and ScanNet, means only, duplicates not removed ==")
    brake = {ds: averages(logged(ds, "ddd3r_brake"))[0] for ds in (TUM, SCANNET)}
    print(f"brake TUM={brake[TUM]:.5f} ScanNet={brake[SCANNET]:.5f}")
    tum_runs = {os.path.basename(p) for p in glob.glob(f"{LOGGED}/{TUM}/*")}
    scannet_runs = {os.path.basename(p) for p in glob.glob(f"{LOGGED}/{SCANNET}/*")}
    shared = sorted(tum_runs & scannet_runs)
    rows, dominating, lower_tum, lower_scan = 0, [], 0, 0
    for name in shared:
        try:
            tum, scan = averages(logged(TUM, name))[0], averages(logged(SCANNET, name))[0]
        except (FileNotFoundError, IndexError):
            continue
        rows += 1
        lower_tum += tum < brake[TUM]
        lower_scan += scan < brake[SCANNET]
        if tum < brake[TUM] and scan < brake[SCANNET]:
            dominating.append((name, tum, scan))
    print(f"runs on both datasets: {rows}; lower on TUM: {lower_tum}; lower on ScanNet: {lower_scan}; lower on both: {len(dominating)}")
    for name, tum, scan in dominating:
        print(f"  {name} TUM={tum:.5f} ScanNet={scan:.5f}")


def main():
    instrument()
    e1()
    e2()
    e3()
    e4()
    reviewer_pairs()
    online_selectors()


if __name__ == "__main__":
    main()
