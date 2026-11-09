"""E9: the drift-energy routing rule on held-out datasets. See docs/PREREG_E9.md.

  python e9_selector.py prefix   GPU. Check I1 on TUM and ScanNet, then compute the
                                 prefix drift energy on KITTI Odometry, Sintel and
                                 ARKitScenes 500f. Writes results/e9_instrument.txt
                                 and results/e9_prefix.json.
  python e9_selector.py rule     CPU. Check I2, route each sequence and rule on P9a
                                 and P9b. Writes results/e9.txt.

Run from the repository root with PYTHONPATH=src.
"""
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from stats_rule import compare, load_ate  # noqa: E402

ROOT = "rebuttal/iclr2027/results"
RELPOSE = f"{ROOT}/relpose"
A4 = "/home/szy/research/TTT3R/analysis_results/a4_drift_scaling"
LOGGED_ARKIT = ("/home/szy/research/TTT3R/.claude/worktrees/understand-full-project-14f204"
                "/rebuttal/results/relpose/arkit_s1_500")
EARLIER_SINTEL_TTT3R = "/home/szy/research/TTT3R/eval_results/relpose/sintel/ttt3r"

THRESHOLD = 0.5
PREFIX = 200
I1_TOL = 0.005
IMAGE_EXT = (".png", ".jpg", ".jpeg")
SINTEL = ["alley_2", "ambush_4", "ambush_5", "ambush_6", "cave_2", "cave_4", "market_2",
          "market_5", "market_6", "shaman_3", "sleeping_1", "sleeping_2", "temple_2", "temple_3"]
I1_SCANNET = ["scene0712_00", "scene0757_00", "scene0794_00", "scene0795_00", "scene0806_00"]


def prefix_files(directory, n=PREFIX):
    """Registered file list: entries sorted by name, image extensions only, first n."""
    names = sorted(os.listdir(directory))
    images = [os.path.join(directory, x) for x in names if x.lower().endswith(IMAGE_EXT)]
    return images[:n], len(images)


def heldout_dirs():
    out = []
    kitti = "rebuttal/iclr2027/data/long_kitti_odo_s1"
    for seq in sorted(d for d in os.listdir(kitti) if os.path.isdir(os.path.join(kitti, d))):
        out.append(("KITTI", seq, os.path.join(kitti, seq, "image_full")))
    for seq in SINTEL:
        out.append(("Sintel", seq, os.path.join("data/sintel/training/final", seq)))
    arkit = "data/long_arkit_s1"
    for seq in sorted(d for d in os.listdir(arkit) if os.path.isdir(os.path.join(arkit, d, "rgb_500"))):
        out.append(("ARKitScenes", seq, os.path.join(arkit, seq, "rgb_500")))
    return out


def prefix():
    sys.path.insert(0, "analysis")
    import a4_drift_scaling as a4

    model = a4.load_model("cuda")

    # I1: reproduce the stored values the rule was chosen on.
    stored = {}
    for name in ("tum", "scannet"):
        for row in json.load(open(f"{A4}/{name}_200f.json"))["per_scene"]:
            stored[(name, row["scene"])] = row["drift_energy_mean"]
    jobs = [("tum", s, f"data/long_tum_s1/{s}/rgb_1000") for s in sorted(os.listdir("data/long_tum_s1"))
            if s.startswith("rgbd")]
    jobs += [("scannet", s, f"data/long_scannet_s3/{s}/color_200") for s in I1_SCANNET]
    lines, ok = [], True
    for name, seq, directory in jobs:
        files, _ = prefix_files(directory)
        e = a4.analyze_scene(model, files)["drift_energy_mean"]
        ref = stored[(name, seq)]
        good = abs(e - ref) <= I1_TOL and ((e < THRESHOLD) == (ref < THRESHOLD))
        ok &= good
        lines.append(f"{name:8s} {seq:44s} n={len(files):3d} stored={ref:.5f} now={e:.5f} "
                     f"diff={e - ref:+.5f} {'ok' if good else 'MISS'}")
        print(lines[-1], flush=True)
    lines.append(f"I1 {'PASS' if ok else 'FAIL'} (tolerance {I1_TOL}, same side of {THRESHOLD})")
    print(lines[-1], flush=True)
    with open(f"{ROOT}/e9_instrument.txt", "w") as handle:
        handle.write("\n".join(lines) + "\n")
    if not ok:
        sys.exit(4)

    rows = []
    for dataset, seq, directory in heldout_dirs():
        files, total = prefix_files(directory)
        stats = a4.analyze_scene(model, files)
        rows.append(dict(dataset=dataset, seq=seq, n=len(files), total_frames=total,
                         drift_energy=stats["drift_energy_mean"], cos_mean=stats["cos_mean"]))
        print(f"{dataset:12s} {seq:12s} n={len(files):3d}/{total:4d} e={stats['drift_energy_mean']:.4f}", flush=True)
        with open(f"{ROOT}/e9_prefix.json", "w") as handle:
            json.dump(rows, handle, indent=1)


_KITTI_ROW = re.compile(r"\s+(\S+)\s+\|\s+(\d\d)\s+\|\s+t_rel:\s+([\d.]+)%\s+\|\s+r_rel:\s+([\d.]+)")


def kitti_rrel(method):
    out = {}
    with open(f"{ROOT}/kitti_official_metrics.txt") as handle:
        for line in handle:
            m = _KITTI_ROW.match(line)
            if m and m.group(1) == method:
                out[m.group(2)] = float(m.group(4))
    return out


def fmt(r):
    lo, hi = r["ci"]
    return (f"{r['mean_a']:.4f} vs {r['mean_b']:.4f} ({100 * r['gap']:+.1f}%) wins={r['wins_a']}/{r['n']} "
            f"CI[{lo:+.4f}, {hi:+.4f}] p={r['p']:.4f} -> {r['verdict']}")


def rule():
    out = []

    def say(text=""):
        out.append(text)
        print(text)

    i1 = open(f"{ROOT}/e9_instrument.txt").read()
    i1_ok = "I1 PASS" in i1
    say(f"== I1 == {'PASS' if i1_ok else 'FAIL'} (results/e9_instrument.txt)")

    # I2 on the new Sintel arms.
    sintel = {a: load_ate(f"{RELPOSE}/sintel/{a}") for a in ("ortho_plain", "brake_t1", "ttt3r_plain")}
    say("\n== I2, Sintel arms ==")
    i2_ok = True
    for arm, values in sintel.items():
        finite = [s for s in SINTEL if s in values and math.isfinite(values[s])]
        good = len(finite) == len(SINTEL)
        i2_ok &= good
        say(f"{arm:12s} finite ATE on {len(finite)}/{len(SINTEL)} sequences {'ok' if good else 'MISS'}")
    for arm in ("ortho_plain", "brake_t1"):
        differ = sum(sintel[arm].get(s) != sintel["ttt3r_plain"].get(s) for s in SINTEL)
        i2_ok &= differ >= 1
        say(f"{arm:12s} differs from ttt3r_plain on {differ}/{len(SINTEL)} sequences {'ok' if differ else 'MISS'}")
    if os.path.exists(f"{EARLIER_SINTEL_TTT3R}/_error_log.txt"):
        earlier = load_ate(EARLIER_SINTEL_TTT3R)
        keys = [s for s in SINTEL if s in earlier and s in sintel["ttt3r_plain"]]
        now = sum(sintel["ttt3r_plain"][s] for s in keys) / len(keys)
        then = sum(earlier[s] for s in keys) / len(keys)
        say(f"ttt3r_plain mean ATE {now:.5f}, earlier run {then:.5f} over {len(keys)} sequences "
            f"({100 * (now / then - 1):+.2f}%), reported only")
    say(f"I2 {'PASS' if i2_ok else 'FAIL'}")

    # Sequence counts on Sintel: prediction length against the image count.
    say("\n== Sequence counts, Sintel ==")
    dropped = set()
    for s in SINTEL:
        _, total = prefix_files(os.path.join("data/sintel/training/final", s), n=10 ** 9)
        for arm in sintel:
            path = f"{RELPOSE}/sintel/{arm}/{s}/pred_traj.txt"
            lines = sum(1 for x in open(path) if x.strip() and not x.startswith("#")) if os.path.exists(path) else -1
            if lines != total:
                dropped.add(s)
                say(f"{s} {arm}: prediction {lines} poses, {total} images, left out")
    if not dropped:
        say("all 14 sequences have one pose per image in every arm")

    # Routing.
    prefix_rows = json.load(open(f"{ROOT}/e9_prefix.json"))
    energy = {(r["dataset"], r["seq"]): r["drift_energy"] for r in prefix_rows}
    say("\n== Prefix drift energy and routing ==")
    for dataset in ("KITTI", "Sintel", "ARKitScenes"):
        rows = [r for r in prefix_rows if r["dataset"] == dataset]
        to_ortho = [r["seq"] for r in rows if r["drift_energy"] < THRESHOLD]
        near = [r["seq"] for r in rows if abs(r["drift_energy"] - THRESHOLD) <= I1_TOL]
        es = [r["drift_energy"] for r in rows]
        say(f"{dataset:12s} n={len(rows):2d} mean e={sum(es) / len(es):.4f} range [{min(es):.4f}, {max(es):.4f}] "
            f"to ortho {len(to_ortho)}/{len(rows)} within {I1_TOL} of {THRESHOLD}: {len(near)}")
        say("   " + "  ".join(f"{r['seq']}={r['drift_energy']:.3f}{'*' if r['drift_energy'] < THRESHOLD else ''}"
                              for r in rows))
    say("(* routed to ortho)")

    cells = {
        ("KITTI", "ATE"): (load_ate(f"{RELPOSE}/kitti_odo_full/ortho_plain"),
                           load_ate(f"{RELPOSE}/kitti_odo_full/brake_t1"),
                           load_ate(f"{RELPOSE}/kitti_odo_full/ttt3r_plain")),
        ("KITTI", "r_rel"): (kitti_rrel("ortho_plain"), kitti_rrel("brake_t1"), kitti_rrel("ttt3r_plain")),
        ("Sintel", "ATE"): (sintel["ortho_plain"], sintel["brake_t1"], sintel["ttt3r_plain"]),
        ("ARKitScenes", "ATE"): (load_ate(f"{RELPOSE}/arkit_s1_500/ortho_check"),
                                 load_ate(f"{LOGGED_ARKIT}/ddd3r_brake"),
                                 load_ate(f"{LOGGED_ARKIT}/ttt3r")),
    }
    say("\n== Cells ==")
    verdicts = {}
    for (dataset, metric), (ortho, brake, ttt3r) in cells.items():
        keys = sorted(s for s in set(ortho) & set(brake) if (dataset, s) in energy
                      and not (dataset == "Sintel" and s in dropped))
        chosen = {s: (ortho[s] if energy[(dataset, s)] < THRESHOLD else brake[s]) for s in keys}
        o = {s: ortho[s] for s in keys}
        b = {s: brake[s] for s in keys}
        branches = compare(o, b)
        informative = branches["verdict"] != "tied"
        better = None if not informative else ("ortho" if branches["verdict"] == "a" else "brake")
        vs_o, vs_b = compare(chosen, o), compare(chosen, b)
        held_out = dataset != "ARKitScenes"
        tag = "held-out" if held_out else "partly seen, not counted"
        say(f"{dataset} {metric} ({tag}), n={len(keys)}, routed to ortho {sum(energy[(dataset, s)] < THRESHOLD for s in keys)}")
        say(f"  ortho vs brake  {fmt(branches)}  informative={informative}")
        say(f"  rule  vs ortho  {fmt(vs_o)}")
        say(f"  rule  vs brake  {fmt(vs_b)}")
        if ttt3r:
            say(f"  rule  vs TTT3R  {fmt(compare(chosen, {s: ttt3r[s] for s in keys if s in ttt3r}))}")
        verdicts[(dataset, metric)] = dict(held_out=held_out, informative=informative, better=better,
                                           vs_o=vs_o["verdict"], vs_b=vs_b["verdict"])

    say("\n== Ruling ==")
    held = {k: v for k, v in verdicts.items() if v["held_out"]}
    informative = [k for k, v in held.items() if v["informative"]]
    p9a = True
    for k in informative:
        v = held[k]
        worse = (v["vs_o"] if v["better"] == "ortho" else v["vs_b"]) == "b"
        p9a &= not worse
        say(f"P9a {k[0]} {k[1]}: better branch {v['better']}, rule {'worse' if worse else 'not worse'}")
    p9b = True
    for k, v in held.items():
        if k[1] == "ATE":
            worse = v["vs_b"] == "b"
            p9b &= not worse
            say(f"P9b {k[0]} ATE: rule {'worse' if worse else 'not worse'} than brake")
    say(f"informative held-out cells: {len(informative)}")
    say(f"P9a {'holds' if p9a else 'FAILS'}; P9b {'holds' if p9b else 'FAILS'}")
    if not (i1_ok and i2_ok):
        say("E9 NO RULING: a check failed. Fix it in a dated correction and rerun what failed")
    else:
        holds = len(informative) >= 1 and p9a and p9b
        say(f"E9 {'HOLDS' if holds else 'FAILS'}")

    with open(f"{ROOT}/e9.txt", "w") as handle:
        handle.write("\n".join(out) + "\n")


if __name__ == "__main__":
    {"prefix": prefix, "rule": rule}[sys.argv[1]]()
