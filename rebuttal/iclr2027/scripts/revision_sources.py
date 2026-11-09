"""Paired tests the revised PDF cites that no earlier results file prints. Not registered.

Run from the repository root:
    python rebuttal/iclr2027/scripts/revision_sources.py > rebuttal/iclr2027/results/revision_sources.txt
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from e1_e4_rulings import LOGGED_ARKIT, averages  # noqa: E402
from e6_rulings import line  # noqa: E402
from stats_rule import load_ate  # noqa: E402


def arkit(name):
    return load_ate(os.path.join(LOGGED_ARKIT, name))


print("== ARKitScenes 500f, appendix text on the operating points ==")
const = arkit("ddd3r_constant")
for other in ("cut3r", "ttt3r", "ddd3r_brake"):
    line(f"arkit const05 vs {other}", const, arkit(other))
line("arkit ortho vs const05", arkit("ddd3r"), const)

print("\n== Gate spread across tokens within a frame, from the E3a gate logs ==")
print("spatial_std: std over tokens in each frame, averaged over frames, then over sequences")
import glob  # noqa: E402

import numpy as np  # noqa: E402

GATES = "rebuttal/iclr2027/results/gates"
for dataset in ("tum_s1_1000", "scannet_s3_1000__first20"):
    for method in ("ttt3r", "ttsa3r"):
        files = sorted(glob.glob(f"{GATES}/{dataset}/{method}/*.npy"))
        spatial = [float(np.load(f).std(axis=1).mean()) for f in files]
        temporal = [float(np.load(f).mean(axis=1).std()) for f in files]
        print(f"  {dataset:26s} {method:7s} n={len(files):2d} spatial_std={np.mean(spatial):.4f} "
              f"frame_mean_std={np.mean(temporal):.4f}")

print("\n== Table 6 gamma rows, ATE from the logged runs ==")
LOGGED_RELPOSE = "/home/szy/research/TTT3R/eval_results/relpose"
for dataset in ("tum_s1_1000", "scannet_s3_1000"):
    for g in ("0.5", "1", "2", "3", "5"):
        ate, _, _ = averages(f"{LOGGED_RELPOSE}/{dataset}/ttt3r_ortho_steep_v2_g{g}")
        print(f"  {dataset:16s} gamma={g:3s} ATE={ate:.5f}")

print("\n== ARKitScenes 500f, mean RPE for the appendix RPE table ==")
for name in ("cut3r", "ttt3r", "ddd3r_constant", "ddd3r_brake", "ddd3r"):
    ate, trans, rot = averages(os.path.join(LOGGED_ARKIT, name))
    print(f"  {name:16s} ATE={ate:.5f} RPE trans={trans:.5f} RPE rot={rot:.5f}")
