"""Subsample ScanNet 90-frame sequences from 1000-frame long sequences.

Source: data/long_scannet_s3/{seq}/color_1000/ + pose_1000.txt
Target: data/long_scannet_s3/{seq}/color_90/ + pose_90.txt

Evenly subsample 90 frames from 1000 to match CUT3R/TTSA3R short-seq protocol.

Usage:
    python datasets_preprocess/prepare_scannet_short.py
"""
import os
import shutil
import numpy as np

SRC_BASE = "data/long_scannet_s3"
TARGET_FRAMES = 90

seqs = sorted([s for s in os.listdir(SRC_BASE) if os.path.isdir(os.path.join(SRC_BASE, s))])
print(f"Found {len(seqs)} ScanNet sequences")

for seq in seqs:
    src_color = os.path.join(SRC_BASE, seq, "color_1000")
    src_pose = os.path.join(SRC_BASE, seq, "pose_1000.txt")

    if not os.path.exists(src_color) or not os.path.exists(src_pose):
        print(f"  Skipping {seq}: missing data")
        continue

    # Get all frames sorted
    frames = sorted(os.listdir(src_color))
    n_total = len(frames)
    if n_total < TARGET_FRAMES:
        print(f"  Skipping {seq}: only {n_total} frames")
        continue

    # Evenly subsample indices
    indices = np.linspace(0, n_total - 1, TARGET_FRAMES, dtype=int)

    # Read poses
    with open(src_pose) as f:
        pose_lines = f.readlines()

    # Create output dirs
    dst_color = os.path.join(SRC_BASE, seq, f"color_{TARGET_FRAMES}")
    os.makedirs(dst_color, exist_ok=True)

    # Copy selected frames
    selected_poses = []
    for idx in indices:
        src_file = os.path.join(src_color, frames[idx])
        dst_file = os.path.join(dst_color, frames[idx])
        if not os.path.exists(dst_file):
            shutil.copy2(src_file, dst_file)
        if idx < len(pose_lines):
            selected_poses.append(pose_lines[idx])

    # Write pose file
    dst_pose = os.path.join(SRC_BASE, seq, f"pose_{TARGET_FRAMES}.txt")
    with open(dst_pose, "w") as f:
        f.writelines(selected_poses)

    print(f"  {seq}: {n_total} -> {len(indices)} frames, {len(selected_poses)} poses")

print("Done.")
