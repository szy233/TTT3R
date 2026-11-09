"""E8: rotation error at long frame gaps (R_Δ) and the ARKitScenes 1000f rulings. See docs/PREREG_E8.md.

R_Δ is the RPE rotation angle (deg, RMSE over all pairs) at a gap of Δ frames, computed with the
same evo call as eval/relpose/evo_utils.py::eval_metrics, which uses Δ=1.

Usage, from the repository root, with the ttt3r environment:
    python rebuttal/iclr2027/scripts/e8_rot_drift.py instrument > rebuttal/iclr2027/results/e8_instrument.txt
    python rebuttal/iclr2027/scripts/e8_rot_drift.py e8a > rebuttal/iclr2027/results/e8a.txt
    python rebuttal/iclr2027/scripts/e8_rot_drift.py e8b > rebuttal/iclr2027/results/e8b.txt
    python rebuttal/iclr2027/scripts/e8_rot_drift.py posthoc > rebuttal/iclr2027/results/e8_posthoc.txt
"""
import math
import os
import re
import sys
from copy import deepcopy

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evo.core import sync  # noqa: E402
from evo.core.filters import FilterException  # noqa: E402
from evo.core.metrics import PoseRelation, Unit  # noqa: E402
from evo.core.trajectory import PoseTrajectory3D  # noqa: E402
from evo.main_rpe import rpe  # noqa: E402
from evo.tools import file_interface  # noqa: E402

from eval.relpose.evo_utils import load_traj, make_traj  # noqa: E402
from eval.relpose.metadata import dataset_metadata  # noqa: E402
from stats_rule import bootstrap, compare, holm, load_ate, wilcoxon  # noqa: E402
from e6_rulings import line, load_rpe  # noqa: E402

RELPOSE = "rebuttal/iclr2027/results/relpose"
GAINS = "rebuttal/iclr2027/results/gains"
ROW = re.compile(r"^(\S+?)\s*\|\s*ATE:\s*[0-9.]+,\s*RPE trans:\s*[0-9.]+,\s*RPE rot:\s*([0-9.]+)")
ARKIT, ARKIT_LONG, TUM, SCANNET, KITTI = "arkit_s1_500", "arkit_s1_1000", "tum_s1_1000", "scannet_s3_1000", "kitti_odo_full"
LONG_GROUP = "arkit_s1_1000__eight"
EIGHT = ["41069021", "41069025", "41069046", "41125696", "41125700", "41125709", "41125718", "41125722"]
GAPS = (100, 50, 200)
SCALES = {ARKIT: ("010", "020", "033"), TUM: ("010", "020", "033"), SCANNET: ("010", "033"), KITTI: ("010", "020", "033")}


def logged_rot(folder):
    """Per-sequence logged RPE rotation at a 1-frame gap, from _error_log.txt."""
    out = {}
    for row in open(os.path.join(folder, "_error_log.txt"), errors="ignore"):
        match = ROW.match(row)
        if match:
            out[match.group(1).split("-", 1)[-1]] = float(match.group(2))
    return out


def trajectories(dataset, folder, seq):
    """Ground truth and prediction, read and matched as eval/relpose/launch.py does."""
    meta = dataset_metadata[dataset]
    gt_file = meta["gt_traj_func"](meta["img_path"], meta.get("anno_path", None), seq)
    gt = make_traj(load_traj(gt_traj_file=gt_file, traj_format=meta["traj_format"]))
    saved = file_interface.read_tum_trajectory_file(os.path.join(folder, seq, "pred_traj.txt"))
    pred = PoseTrajectory3D(
        positions_xyz=saved.positions_xyz, orientations_quat_wxyz=saved.orientations_quat_wxyz, timestamps=saved.timestamps
    )
    if pred.timestamps.shape[0] != gt.timestamps.shape[0]:
        return None
    pred.timestamps = gt.timestamps
    return sync.associate_trajectories(gt, pred)


def r_gap(pair, gap):
    gt, pred = pair
    result = rpe(
        deepcopy(gt), deepcopy(pred), est_name="traj", pose_relation=PoseRelation.rotation_angle_deg,
        align=True, correct_scale=True, delta=gap, delta_unit=Unit.frames, rel_delta_tol=0.01, all_pairs=True,
    )
    return result.stats["rmse"]


_CACHE = {}


def load_r(dataset, folder, gaps=GAPS, only=None):
    """{gap: {seq: R_gap}} over the sequences the run logged. Sequences with a length mismatch are reported and left out."""
    key = (folder, gaps, tuple(only or ()))
    if key in _CACHE:
        return _CACHE[key]
    out = {gap: {} for gap in gaps}
    for seq in sorted(logged_rot(folder)):
        if only and seq not in only:
            continue
        pair = trajectories(dataset, folder, seq)
        if pair is None:
            print(f"  length mismatch, left out: {folder}/{seq}")
            continue
        for gap in gaps:
            try:
                out[gap][seq] = r_gap(pair, gap)
            except FilterException:
                # Deviation recorded in PREREG_E8.md: the sequence has too few frames for this gap.
                print(f"  too short for a {gap}-frame gap, left out of R_{gap}: {folder}/{seq}")
    _CACHE[key] = out
    return out


def arm(dataset, name, group=None):
    return f"{RELPOSE}/{group or dataset}/{name}"


def instrument():
    print("== I1, R_1 recomputed against the logged per-sequence RPE rotation (tolerance 0.5%) ==")
    worst_all = 0.0
    for dataset, name in ((ARKIT, "dir010"), (TUM, "ortho_check"), (SCANNET, "ortho_check"), (KITTI, "ortho_check")):
        folder = arm(dataset, name)
        logged = logged_rot(folder)
        mine = load_r(dataset, folder, gaps=(1,))[1]
        gaps = [abs(mine[k] - logged[k]) / max(logged[k], 1e-12) for k in mine]
        worst = max(gaps)
        worst_all = max(worst_all, worst)
        print(f"  {dataset:16s} {name:12s} n={len(mine):2d} of {len(logged):2d} logged, largest gap={100 * worst:.4f}%")
    print(f"I1 {'passes' if worst_all <= 0.005 else 'MISSES'}")


def designs(dataset, group=None, only=None):
    """Yield (label, direction arm folder, reference arm folder)."""
    yield "A", arm(dataset, "ortho_check", group), arm(dataset, "normreplay", group)
    for scale in SCALES[dataset] if group is None else ("010", "033"):
        yield f"B{scale}", arm(dataset, f"dir{scale}", group), arm(dataset, f"const{scale}", group)


def r_block(dataset, group=None, only=None, meta_key=None):
    """R_Δ tests for both designs. Returns {label: compare result at R_100}."""
    meta_key = meta_key or dataset
    results, pvalues = {}, []
    for gap in GAPS:
        print(f"R_{gap}:")
        for label, direction, reference in designs(dataset, group, only):
            a = load_r(meta_key, direction, only=only)[gap]
            b = load_r(meta_key, reference, only=only)[gap]
            r = line(f"  {label} {os.path.basename(direction)} vs {os.path.basename(reference)}", a, b)
            if gap == 100:
                results[label] = r
                pvalues.append(r["p"])
    adjusted = holm(pvalues)
    print("  Holm-adjusted p at R_100: " + ", ".join(f"{k}={p:.4f}" for k, p in zip(results, adjusted)))
    return results


def noise(dataset, group=None, meta_key=None):
    meta_key = meta_key or dataset
    a = load_r(meta_key, arm(dataset, "orthop", group))[100]
    b = load_r(meta_key, arm(dataset, "ortho_check", group))[100]
    keys = sorted(set(a) & set(b))
    ma, mb = sum(a[k] for k in keys) / len(keys), sum(b[k] for k in keys) / len(keys)
    print(f"  rounding noise R_100 {dataset}: orthop {ma:.4f} vs ortho {mb:.4f} ({100 * (ma / mb - 1):+.1f}%)")
    return abs(ma / mb - 1)


def e8a():
    rulings = {}
    for dataset, role in ((KITTI, "instrument I2"), (ARKIT, "primary"), (TUM, "secondary"), (SCANNET, "secondary")):
        print(f"\n== {dataset} ({role}) ==")
        rulings[dataset] = r_block(dataset)
    print("\n== Rounding noise ==")
    noise(TUM)
    noise(KITTI)

    print("\n== I2: R_100 sign on KITTI, where r_rel says direction is lower in all four cells ==")
    signs = {k: r["mean_a"] < r["mean_b"] for k, r in rulings[KITTI].items()}
    print("  " + ", ".join(f"{k}: direction {'lower' if v else 'higher'}" for k, v in signs.items()))
    i2 = all(signs.values()) and len(signs) == 4
    print(f"I2 {'passes' if i2 else 'fails'}")

    ark = rulings[ARKIT]
    p8a = ark["A"]["verdict"] == "a"
    p8b = sum(ark[k]["verdict"] == "a" for k in ("B010", "B020", "B033")) >= 2
    print(f"\nP8a (ARKit 500 R_100, design A better): {'holds' if p8a else 'fails'} ({ark['A']['verdict']})")
    print(f"P8b (ARKit 500 R_100, design B better at >= 2 of 3): {'holds' if p8b else 'fails'} "
          f"({', '.join(ark[k]['verdict'] for k in ('B010', 'B020', 'B033'))})")
    general = (
        all(rulings[d]["A"]["verdict"] == "a" for d in (ARKIT, TUM, SCANNET, KITTI))
        and sum(rulings[ARKIT][k]["verdict"] == "a" for k in ("B010", "B020", "B033")) >= 2
        and sum(rulings[TUM][k]["verdict"] == "a" for k in ("B010", "B020", "B033")) >= 2
        and all(rulings[SCANNET][k]["verdict"] == "a" for k in ("B010", "B033"))
    )
    print(f"General four-dataset statement on R_100: {'allowed' if general else 'not allowed'}")


def check_gain(name, scale):
    folder = f"{GAINS}/{LONG_GROUP}/{name}"
    if not os.path.isdir(folder):
        print(f"  I3 {name}: no gain log")
        return
    import numpy as np
    worst = 0.0
    files = [f for f in os.listdir(folder) if f.endswith(".npy")]
    for f in files:
        g = np.load(os.path.join(folder, f))
        worst = max(worst, abs(float(np.mean(g)) - scale) / scale)
    print(f"  I3 {name}: {len(files)} logs, largest gap {100 * worst:.4f}% {'pass' if worst <= 0.005 else 'MISS'}")


def e8b():
    print("== I3, gain of the dir arms at 1000f ==")
    check_gain("dir033", 0.33)
    check_gain("dir010", 0.10)

    print("\n== ARKitScenes 1000f, 8 scenes, ATE RMSE ==")
    ate = {}
    for label, direction, reference in designs(ARKIT_LONG, LONG_GROUP):
        ate[label] = (load_ate(direction), load_ate(reference))
        line(f"  {label} {os.path.basename(direction)} vs {os.path.basename(reference)}", *ate[label])
    ortho = load_ate(arm(ARKIT_LONG, "ortho_check", LONG_GROUP))
    print("Against the default ortho, no prediction:")
    for name in ("normreplay", "dir033", "const033", "dir010", "const010", "ttt3r_plain", "brake_t1"):
        folder = arm(ARKIT_LONG, name, LONG_GROUP)
        if os.path.exists(os.path.join(folder, "_error_log.txt")):
            line(f"  {name} vs ortho_check", load_ate(folder), ortho)
    o, p = load_ate(arm(ARKIT_LONG, "orthop", LONG_GROUP)), ortho
    keys = sorted(set(o) & set(p))
    mo, mp = sum(o[k] for k in keys) / len(keys), sum(p[k] for k in keys) / len(keys)
    print(f"  rounding noise ATE ARKit 1000: orthop {mo:.5f} vs ortho {mp:.5f} ({100 * (mo / mp - 1):+.1f}%)")

    print("\n== ARKitScenes 1000f, 8 scenes, RPE at a 1-frame gap ==")
    for label, direction, reference in designs(ARKIT_LONG, LONG_GROUP):
        (t_d, r_d), (t_r, r_r) = load_rpe(direction), load_rpe(reference)
        line(f"  {label} RPE trans", t_d, t_r)
        line(f"  {label} RPE rot", r_d, r_r)

    print("\n== ARKitScenes 1000f, 8 scenes, R_Δ ==")
    long_r = r_block(ARKIT_LONG, LONG_GROUP)
    noise(ARKIT_LONG, LONG_GROUP)
    p8c = all(long_r[k]["verdict"] == "a" for k in ("A", "B010", "B033"))
    print(f"P8c (ARKit 1000 R_100 better in A, B010 and B033): {'holds' if p8c else 'fails'} "
          f"({', '.join(long_r[k]['verdict'] for k in ('A', 'B010', 'B033'))})")

    print("\n== P8d, change of ln(ATE ratio) from 500 to 1000 frames on the 8 scenes ==")
    supported = 0
    for label, direction, reference in designs(ARKIT_LONG, LONG_GROUP):
        short_dir = load_ate(arm(ARKIT, os.path.basename(direction)))
        short_ref = load_ate(arm(ARKIT, os.path.basename(reference)))
        long_dir, long_ref = ate[label]
        keys = [k for k in EIGHT if k in long_dir and k in long_ref and k in short_dir and k in short_ref]
        d = [math.log(long_dir[k] / long_ref[k]) - math.log(short_dir[k] / short_ref[k]) for k in keys]
        mean = sum(d) / len(d)
        lo, hi = bootstrap(d)
        p = wilcoxon(d)
        r500 = sum(math.log(short_dir[k] / short_ref[k]) for k in keys) / len(keys)
        r1000 = sum(math.log(long_dir[k] / long_ref[k]) for k in keys) / len(keys)
        ok = mean < -0.022 and p < 0.05
        supported += ok
        print(f"  {label}: n={len(keys)} mean ln ratio 500f {r500:+.4f}, 1000f {r1000:+.4f}, change {mean:+.4f} "
              f"CI[{lo:+.4f},{hi:+.4f}] p={p:.4f} -> {'supports' if ok else 'does not support'}")
    print(f"P8d: {'holds' if supported >= 2 else 'fails'} ({supported} of 3 cells)")
    better = sum(compare(*ate[k])["verdict"] == "a" for k in ("A", "B010", "B033"))
    print(f"Direction better on ARKit 1000 ATE in {better} of 3 cells")


def posthoc():
    """Not registered. ARKitScenes 1000f arms against TTT3R, for the replies' "What stands" sentence."""
    print("== Post hoc, not registered: ARKitScenes 1000f, 8 scenes, against ttt3r_plain ==")
    ttt3r = arm(ARKIT_LONG, "ttt3r_plain", LONG_GROUP)
    for name in ("brake_t1", "const033", "const010", "ortho_check", "normreplay"):
        line(f"  {name} vs ttt3r_plain ATE", load_ate(arm(ARKIT_LONG, name, LONG_GROUP)), load_ate(ttt3r))
    for name in ("brake_t1", "ortho_check"):
        line(f"  {name} vs ttt3r_plain RPE rot", load_rpe(arm(ARKIT_LONG, name, LONG_GROUP))[1], load_rpe(ttt3r)[1])


if __name__ == "__main__":
    {"instrument": instrument, "e8a": e8a, "e8b": e8b, "posthoc": posthoc}[sys.argv[1]]()
