"""Post hoc comparisons read after the E7 rulings. None of these is registered.

Usage, from the repository root:
    python3 rebuttal/iclr2027/scripts/e7_posthoc.py > rebuttal/iclr2027/results/e7_posthoc.txt
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e6_rulings import ARKIT, KITTI, OTHER, RESULTS, SCANNET, TUM, arm, line, load_rpe  # noqa: E402
from kitti_e5b import load_official  # noqa: E402
from stats_rule import load_ate  # noqa: E402


def main():
    t6, _ = load_official(f"{RESULTS}/e6_kitti_metrics.txt")
    t7, _ = load_official(f"{RESULTS}/e7_kitti_metrics.txt")
    translation = {**t6, **t7}

    print("== KITTI t_rel under the Amendment 1 rounding perturbation ==")
    for a, b in (("orthop", "ortho_check"), ("const033p", "const033")):
        line(f"t_rel {a} vs {b}", translation[a], translation[b])

    print("\n== Ortho against const 0.10 beside the primary metrics ==")
    line("KITTI ATE ortho vs const010", load_ate(arm(KITTI, "ortho_check")), load_ate(arm(KITTI, "const010")))
    line("KITTI t_rel ortho vs const010", translation["ortho_check"], translation["const010"])
    for dataset in (TUM, SCANNET, ARKIT, KITTI):
        _, r_ortho = load_rpe(arm(dataset, "ortho_check"))
        _, r_flat = load_rpe(arm(dataset, "const010"))
        line(f"{dataset} RPE rot ortho vs const010", r_ortho, r_flat)

    print("\n== Const 0.33 on ARKitScenes ==")
    const = load_ate(arm(ARKIT, "const033"))
    for other in ("ddd3r_brake", "ttt3r", "ttsa3r", "ddd3r_constant"):
        line(f"ARKit const033 vs {other}", const, load_ate(f"{OTHER}/{ARKIT}/{other}"))

    print("\n== Const 0.10 against const 0.33 on ScanNet ==")
    line("ScanNet const010 vs const033", load_ate(arm(SCANNET, "const010")), load_ate(arm(SCANNET, "const033")))


if __name__ == "__main__":
    main()
