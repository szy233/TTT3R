"""E6 tables from the logs. Runs no model, and prints whatever has finished so far.
Rulings are written by hand in docs/PREREG_E6.md.

Usage, from the repository root:
    bash rebuttal/iclr2027/scripts/e6_kitti_metrics.sh
    python3 rebuttal/iclr2027/scripts/e6_rulings.py > rebuttal/iclr2027/results/e6.txt
"""
import hashlib
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kitti_e5b import load_official  # noqa: E402
from stats_rule import compare, gap_shrink, holm, load_ate  # noqa: E402

RESULTS = "rebuttal/iclr2027/results"
RELPOSE = f"{RESULTS}/relpose"
LOGGED = "/home/szy/research/TTT3R/eval_results/relpose"
OTHER = "/home/szy/research/TTT3R/.claude/worktrees/understand-full-project-14f204/rebuttal/results/relpose"
TUM, SCANNET, ARKIT, KITTI = "tum_s1_1000", "scannet_s3_1000", "arkit_s1_500", "kitti_odo_full"
FLAT = ["const010", "const015", "const020", "const025", "const033"]
AVERAGE = re.compile(r"Average ATE:\s*([0-9.]+),\s*Average RPE trans:\s*([0-9.]+),\s*Average RPE rot:\s*([0-9.]+)")
ROW = re.compile(r"^(\S+?)\s*\|\s*ATE:\s*[0-9.]+,\s*RPE trans:\s*([0-9.]+),\s*RPE rot:\s*([0-9.]+)")
EXPECTED = {KITTI: 11, TUM: 8, SCANNET: 65, ARKIT: 18}
BLOWUP = "rgbd_dataset_freiburg3_sitting_halfsphere"  # Flat α up to 0.25 reach 0.13 to 0.24 here, ortho 0.062.


def without(values, key):
    return {k: v for k, v in values.items() if k != key}


def arm(dataset, name):
    return f"{RELPOSE}/{dataset}/{name}"


def finished(folder):
    """True once the run has written its final average."""
    try:
        with open(os.path.join(folder, "_error_log.txt"), errors="ignore") as handle:
            return any(AVERAGE.search(line) for line in handle)
    except FileNotFoundError:
        return False


def averages(folder):
    with open(os.path.join(folder, "_error_log.txt"), errors="ignore") as handle:
        found = [AVERAGE.search(line) for line in handle]
    return tuple(float(x) for x in [m for m in found if m][-1].groups())


def load_rpe(folder):
    trans, rot = {}, {}
    with open(os.path.join(folder, "_error_log.txt"), errors="ignore") as handle:
        for line in handle:
            match = ROW.match(line)
            if match:
                key = match.group(1).split("-", 1)[-1]
                trans[key], rot[key] = float(match.group(2)), float(match.group(3))
    return trans, rot


def line(label, a, b):
    r = compare(a, b)
    print(
        f"{label:46s} n={r['n']:2d} {r['mean_a']:.5f} vs {r['mean_b']:.5f} ({100 * r['gap']:+6.1f}%) "
        f"wins={r['wins_a']:2d}/{r['n']} CI[{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}] p={r['p']:.4f} -> {r['verdict']}"
    )
    return r


def rpe(label, folder):
    ate, trans, rot = averages(folder)
    print(f"  {label:24s} ATE={ate:.5f} RPE trans={trans:.5f} RPE rot={rot:.5f}")


def require(dataset, *names):
    missing = [n for n in names if not finished(arm(dataset, n))]
    if missing:
        print(f"waiting for {dataset}: {', '.join(missing)}")
    return not missing


def check_count(dataset, name):
    n = len(load_ate(arm(dataset, name)))
    if n != EXPECTED[dataset]:
        print(f"WARNING {dataset}/{name} has {n} sequences, expected {EXPECTED[dataset]}")


def registration(document, record):
    with open(f"rebuttal/iclr2027/docs/{document}", "rb") as handle:
        # The registered text ends at the first line holding only "---". Dated results follow it.
        body = handle.read().split(b"\n---\n")[0] + b"\n---\n"
    with open(f"{RESULTS}/{record}") as handle:
        recorded = handle.read().split()
    print(f"{document}: registered {recorded[0]} on {' '.join(recorded[2:])}")
    print(f"{document}: file now   {hashlib.sha256(body).hexdigest()}")


def e6a():
    print("\n== E6a, KITTI same norm ==")
    if not require(KITTI, "ortho_check", "normreplay"):
        return
    fresh, plain = load_ate(arm(KITTI, "ortho_check")), load_ate(arm(KITTI, "ortho_plain"))
    worst = max(abs(fresh[k] - plain[k]) / plain[k] for k in plain)
    print(f"check: ortho_check vs ortho_plain n={len(plain)} largest gap={100 * worst:.4f}% ({'pass' if worst <= 0.005 else 'MISS'})")
    t_new, r_new = load_official(f"{RESULTS}/e6_kitti_metrics.txt")
    t_old, r_old = load_official(f"{RESULTS}/kitti_official_metrics.txt")
    print("Primary, r_rel:")
    line("r_rel ortho vs normreplay", r_new["ortho_check"], r_new["normreplay"])
    print("Beside it, no prediction:")
    line("t_rel ortho vs normreplay", t_new["ortho_check"], t_new["normreplay"])
    line("ATE RMSE ortho vs normreplay", fresh, load_ate(arm(KITTI, "normreplay")))
    trans_ortho, rot_ortho = load_rpe(arm(KITTI, "ortho_check"))
    trans_replay, rot_replay = load_rpe(arm(KITTI, "normreplay"))
    line("RPE trans ortho vs normreplay", trans_ortho, trans_replay)
    line("RPE rot ortho vs normreplay", rot_ortho, rot_replay)
    print("r_rel per sequence, and with that sequence left out, not registered:")
    ortho_rel, replay_rel = r_new["ortho_check"], r_new["normreplay"]
    for left_out in sorted(ortho_rel):
        print(f"  {left_out} {ortho_rel[left_out]:7.3f} vs {replay_rel[left_out]:7.3f}", end="  ")
        line(
            "without it",
            {k: v for k, v in ortho_rel.items() if k != left_out},
            {k: v for k, v in replay_rel.items() if k != left_out},
        )
    print("Same norm against the scalars, not registered:")
    for name in ("brake_t1", "const05", "ttt3r_plain"):
        line(f"r_rel normreplay vs {name}", r_new["normreplay"], r_old[name])
        line(f"ATE RMSE normreplay vs {name}", load_ate(arm(KITTI, "normreplay")), load_ate(arm(KITTI, name)))
    for name in ("ortho_check", "normreplay", "ttt3r_plain", "const05", "brake_t1"):
        rpe(name, arm(KITTI, name))
    print(f"check: r_rel ortho_check vs ortho_plain largest gap="
          f"{100 * max(abs(r_new['ortho_check'][k] - r_old['ortho_plain'][k]) / r_old['ortho_plain'][k] for k in r_old['ortho_plain']):.4f}%")


def e6b():
    print("\n== E6b, flat multipliers on TUM. a=ortho_check ==")
    done = [name for name in FLAT if finished(arm(TUM, name))]
    if len(done) < len(FLAT):
        print(f"waiting for: {', '.join(n for n in FLAT if n not in done)}")
    if not done:
        return
    ortho = load_ate(arm(TUM, "ortho_check"))
    results = [line(f"ortho vs {name}", ortho, load_ate(arm(TUM, name))) for name in done]
    line("ortho vs const05 (logged, not new)", ortho, load_ate(f"{LOGGED}/{TUM}/ddd3r_constant"))
    line("ortho vs brake (logged, not new)", ortho, load_ate(f"{LOGGED}/{TUM}/ddd3r_brake"))
    print("Holm over the new tests: " + " ".join(f"{n}={p:.4f}" for n, p in zip(done, holm([r["p"] for r in results]))))
    print("Means:")
    for name in done:
        rpe(name, arm(TUM, name))
    rpe("ortho_check", arm(TUM, "ortho_check"))
    print(f"lowest flat mean: {min(done, key=lambda n: averages(arm(TUM, n))[0])}")
    if "const033" in done:
        fresh = load_ate(arm(TUM, "const033"))
        logged_brake = load_ate(f"{LOGGED}/{TUM}/ddd3r_brake")
        line("const033 fresh vs older-code run", fresh, load_ate(f"{LOGGED}/{TUM}/ttt3r_random_p033"))
        print("Not registered:")
        line("const033 vs brake (logged)", fresh, logged_brake)
        line("normreplay vs const033", load_ate(arm(TUM, "normreplay")), fresh)
        if finished(arm(TUM, "brake_t1")):
            fresh_brake = load_ate(arm(TUM, "brake_t1"))
            line("brake_t1 fresh vs brake (logged)", fresh_brake, logged_brake)
            line("const033 vs brake_t1, both fresh", fresh, fresh_brake)
        else:
            print("waiting for tum_s1_1000/brake_t1")
    print(f"Without {BLOWUP}, not registered:")
    for name in done:
        line(f"ortho vs {name}", without(ortho, BLOWUP), without(load_ate(arm(TUM, name)), BLOWUP))


def e6c():
    print("\n== E6c, alpha_parallel=0.15 ==")
    print("Instrument: ortho_a015 against the logged ddd3r_a15 folders, tolerance 2.2%")
    for dataset, logged in ((TUM, 0.05504), (SCANNET, 0.39886)):
        if finished(arm(dataset, "ortho_a015")):
            check_count(dataset, "ortho_a015")
            mean = averages(arm(dataset, "ortho_a015"))[0]
            gap = abs(mean - logged) / logged
            print(f"  {dataset} {mean:.5f} logged {logged} gap={100 * gap:.3f}% {'pass' if gap <= 0.022 else 'MISS'}")
        else:
            print(f"  waiting for {dataset}/ortho_a015")

    print("\nP6c1, TUM")
    if require(TUM, "ortho_a015", "normreplay_a015"):
        line("TUM ortho_a015 vs normreplay_a015", load_ate(arm(TUM, "ortho_a015")), load_ate(arm(TUM, "normreplay_a015")))
        for name in ("ortho_a015", "normreplay_a015"):
            rpe(name, arm(TUM, name))

    print("\nP6c2, ScanNet")
    if require(SCANNET, "ortho_a015", "normreplay_a015"):
        o15, r15 = load_ate(arm(SCANNET, "ortho_a015")), load_ate(arm(SCANNET, "normreplay_a015"))
        o05, r05 = load_ate(arm(SCANNET, "ortho_check")), load_ate(arm(SCANNET, "normreplay"))
        line("ScanNet ortho_a015 vs normreplay_a015", o15, r15)
        mean, (lo, hi) = gap_shrink(o05, r05, o15, r15)
        print(f"harm at 0.05 minus harm at 0.15: mean={mean:+.4f} CI[{lo:+.4f},{hi:+.4f}] -> {'smaller' if lo > 0 else 'not shown smaller'}")
        line("ScanNet normreplay_a015 vs brake (logged)", r15, load_ate(f"{LOGGED}/{SCANNET}/ddd3r_brake"))
        for name in ("ortho_a015", "normreplay_a015"):
            rpe(name, arm(SCANNET, name))

    print("\nP6c3, ARKitScenes, out of sample")
    if require(ARKIT, "ortho_a015", "normreplay_a015"):
        check_count(ARKIT, "ortho_a015")
        o15 = load_ate(arm(ARKIT, "ortho_a015"))
        line("ARKit ortho_a015 vs ortho_check", o15, load_ate(arm(ARKIT, "ortho_check")))
        line("ARKit ortho_a015 vs ttt3r", o15, load_ate(f"{OTHER}/{ARKIT}/ttt3r"))
        line("ARKit ortho_a015 vs ttsa3r", o15, load_ate(f"{OTHER}/{ARKIT}/ttsa3r"))
        line("ARKit ortho_a015 vs brake, not registered", o15, load_ate(f"{OTHER}/{ARKIT}/ddd3r_brake"))
        line("ARKit ortho_a015 vs normreplay_a015, not registered", o15, load_ate(arm(ARKIT, "normreplay_a015")))
        for name in ("ortho_a015", "normreplay_a015"):
            rpe(name, arm(ARKIT, name))

    print("\nP6c4, KITTI, out of sample")
    if require(KITTI, "ortho_a015"):
        check_count(KITTI, "ortho_a015")
        _, r_new = load_official(f"{RESULTS}/e6_kitti_metrics.txt")
        _, r_old = load_official(f"{RESULTS}/kitti_official_metrics.txt")
        line("r_rel ortho_a015 vs brake_t1", r_new["ortho_a015"], r_old["brake_t1"])
        line("r_rel ortho_a015 vs ortho_plain, not registered", r_new["ortho_a015"], r_old["ortho_plain"])
        o15 = load_ate(arm(KITTI, "ortho_a015"))
        line("ATE RMSE ortho_a015 vs brake_t1", o15, load_ate(arm(KITTI, "brake_t1")))
        line("ATE RMSE ortho_a015 vs ortho_plain", o15, load_ate(arm(KITTI, "ortho_plain")))
        rpe("ortho_a015", arm(KITTI, "ortho_a015"))


def e6d():
    print("\n== E6d, flat multipliers on KITTI. a=ortho_check ==")
    missing = [name for name in FLAT if not finished(arm(KITTI, name))]
    if missing:
        print(f"waiting for: {', '.join(missing)}")
    t_rel, r_rel = load_official(f"{RESULTS}/e6_kitti_metrics.txt")
    _, r_old = load_official(f"{RESULTS}/kitti_official_metrics.txt")
    done = [name for name in FLAT if name in r_rel and name not in missing]
    if not done:
        return
    ate = {name: load_ate(arm(KITTI, name)) for name in done + ["ortho_check"]}
    if "const033" in done:
        for seq, logged in (("03", 60.68427), ("04", 5.09581)):
            gap = abs(ate["const033"][seq] - logged) / logged
            print(f"check: const033 seq{seq} {ate['const033'][seq]:.5f} logged {logged} gap={100 * gap:.3f}% {'pass' if gap <= 0.005 else 'MISS'}")
    print("Primary, r_rel:")
    results = [line(f"r_rel ortho vs {name}", r_rel["ortho_check"], r_rel[name]) for name in done]
    line("r_rel ortho vs const05 (E5b, not new)", r_rel["ortho_check"], r_old["const05"])
    print("Holm over the new tests: " + " ".join(f"{n}={p:.4f}" for n, p in zip(done, holm([r["p"] for r in results]))))
    print(f"lowest flat r_rel: {min(done, key=lambda n: sum(r_rel[n].values()))}")
    print("Beside it, no prediction:")
    trans_ortho, rot_ortho = load_rpe(arm(KITTI, "ortho_check"))
    for name in done:
        check_count(KITTI, name)
        trans, rot = load_rpe(arm(KITTI, name))
        line(f"t_rel ortho vs {name}", t_rel["ortho_check"], t_rel[name])
        line(f"ATE RMSE ortho vs {name}", ate["ortho_check"], ate[name])
        line(f"RPE trans ortho vs {name}", trans_ortho, trans)
        line(f"RPE rot ortho vs {name}", rot_ortho, rot)
    print("Means:")
    for name in done + ["ortho_check"]:
        rpe(name, arm(KITTI, name))


def main():
    registration("PREREG_E6.md", "e6_registration.sha256")
    registration("PREREG_E6D.md", "e6d_registration.sha256")
    e6a()
    e6b()
    e6c()
    e6d()


if __name__ == "__main__":
    main()
