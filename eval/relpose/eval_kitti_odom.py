"""Evaluate saved trajectories using standard KITTI odometry metrics (t_rel %, r_rel deg/100m)."""

import os
import sys
import argparse
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from eval.relpose.evo_utils import (
    load_replica_traj,
    kitti_odom_eval_from_tum,
)
from evo.tools import file_interface


def load_pred_traj_tum(pred_file):
    """Load predicted trajectory in TUM format."""
    traj = file_interface.read_tum_trajectory_file(pred_file)
    xyz = traj.positions_xyz
    quat = traj.orientations_quat_wxyz
    timestamps = traj.timestamps
    traj_tum = np.column_stack((xyz, quat))
    return traj_tum, timestamps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Base dir containing method subdirs (cut3r/, ttt3r/, etc.)")
    parser.add_argument("--gt_dir", type=str, default="/mnt/sda/rkj/VGGT_Long_outdoor/kitti_00_10/gt_poses_kitti",
                        help="Directory containing GT pose files (00.txt, 01.txt, ...)")
    parser.add_argument("--seqs", nargs="+", default=None,
                        help="Sequences to evaluate (e.g., 03 04). Auto-detect if not specified.")
    args = parser.parse_args()

    # Auto-detect methods
    methods = sorted([
        d for d in os.listdir(args.results_dir)
        if os.path.isdir(os.path.join(args.results_dir, d)) and not d.startswith("_")
    ])

    # Auto-detect sequences from first method
    if args.seqs is None:
        first_method_dir = os.path.join(args.results_dir, methods[0])
        args.seqs = sorted([
            d for d in os.listdir(first_method_dir)
            if os.path.isdir(os.path.join(first_method_dir, d))
        ])

    print(f"Methods: {methods}")
    print(f"Sequences: {args.seqs}")
    print()

    # Evaluate
    all_results = {}
    for method in methods:
        method_t = []
        method_r = []
        for seq in args.seqs:
            pred_file = os.path.join(args.results_dir, method, seq, "pred_traj.txt")
            gt_file = os.path.join(args.gt_dir, f"{seq}.txt")

            if not os.path.exists(pred_file):
                print(f"  SKIP {method}/{seq}: pred_traj.txt not found")
                continue
            if not os.path.exists(gt_file):
                print(f"  SKIP {method}/{seq}: GT not found")
                continue

            pred_traj = load_pred_traj_tum(pred_file)
            gt_traj = load_replica_traj(gt_file)

            t_rel, r_rel, by_length = kitti_odom_eval_from_tum(pred_traj, gt_traj)

            method_t.append(t_rel)
            method_r.append(r_rel)

            print(f"  {method:12s} | {seq} | t_rel: {t_rel:7.2f}% | r_rel: {r_rel:7.3f} deg/100m | path: {sum(d['count'] for d in by_length.values())} pairs")
            for length, res in sorted(by_length.items()):
                print(f"    {length:4d}m: t={res['t_rel']:6.2f}%  r={res['r_rel']:6.3f} deg/100m  ({res['count']} pairs)")

        if method_t:
            all_results[method] = {
                "t_rel": np.mean(method_t),
                "r_rel": np.mean(method_r),
            }

    # Summary table
    print("\n" + "=" * 65)
    print(f"{'Method':12s} | {'t_rel (%)':>10s} | {'r_rel (deg/100m)':>16s}")
    print("-" * 65)
    baseline_t = all_results.get("cut3r", {}).get("t_rel", None)
    for method in methods:
        if method in all_results:
            t = all_results[method]["t_rel"]
            r = all_results[method]["r_rel"]
            if baseline_t and method != "cut3r":
                improv = (t - baseline_t) / baseline_t * 100
                print(f"{method:12s} | {t:10.2f} | {r:16.3f} | {improv:+.1f}%")
            else:
                print(f"{method:12s} | {t:10.2f} | {r:16.3f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
