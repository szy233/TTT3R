"""Generate missing frame-count configs from existing 1000f data.

For ScanNet: subsample from color_1000 (stride-3 already applied, ~260 frames per scene)
For TUM: subsample from rgb_1000 (stride-1, up to 1000 frames per scene)

Protocol: take the FIRST N frames (sorted), matching TTT3R's long_prepare_*.py behavior.
"""
import os
import shutil
import glob

# === ScanNet ===
SCANNET_BASE = "data/long_scannet_s3"
SCANNET_ALL = [50, 90, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]

scannet_seqs = sorted([s for s in os.listdir(SCANNET_BASE) if os.path.isdir(os.path.join(SCANNET_BASE, s))])
print(f"ScanNet: {len(scannet_seqs)} sequences")

for seq in scannet_seqs:
    src_color = os.path.join(SCANNET_BASE, seq, "color_1000")
    src_pose = os.path.join(SCANNET_BASE, seq, "pose_1000.txt")
    if not os.path.exists(src_color) or not os.path.exists(src_pose):
        print(f"  Skip {seq}: missing color_1000 or pose_1000.txt")
        continue

    all_frames = sorted(os.listdir(src_color))
    with open(src_pose) as f:
        all_poses = f.readlines()

    n_total = len(all_frames)

    for N in SCANNET_ALL:
        dst_color = os.path.join(SCANNET_BASE, seq, f"color_{N}")
        dst_pose = os.path.join(SCANNET_BASE, seq, f"pose_{N}.txt")

        if os.path.exists(dst_color) and os.path.exists(dst_pose):
            continue  # already exists

        actual_n = min(N, n_total)
        selected_frames = all_frames[:actual_n]
        selected_poses = all_poses[:actual_n]

        os.makedirs(dst_color, exist_ok=True)
        for fname in selected_frames:
            src = os.path.join(src_color, fname)
            dst = os.path.join(dst_color, fname)
            if not os.path.exists(dst):
                os.symlink(os.path.abspath(src), dst)

        with open(dst_pose, "w") as f:
            f.writelines(selected_poses)

        print(f"  {seq}: color_{N} -> {actual_n} frames (symlinked)")

# === TUM ===
TUM_BASE = "data/long_tum_s1"
TUM_ALL = [50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

tum_seqs = sorted([s for s in os.listdir(TUM_BASE) if os.path.isdir(os.path.join(TUM_BASE, s))])
print(f"\nTUM: {len(tum_seqs)} sequences")

for seq in tum_seqs:
    src_rgb = os.path.join(TUM_BASE, seq, "rgb_1000")
    src_gt = os.path.join(TUM_BASE, seq, "groundtruth_1000.txt")
    if not os.path.exists(src_rgb) or not os.path.exists(src_gt):
        print(f"  Skip {seq}: missing rgb_1000 or groundtruth_1000.txt")
        continue

    all_frames = sorted(os.listdir(src_rgb))
    with open(src_gt) as f:
        all_poses = f.readlines()

    n_total = len(all_frames)

    for N in TUM_ALL:
        dst_rgb = os.path.join(TUM_BASE, seq, f"rgb_{N}")
        dst_gt = os.path.join(TUM_BASE, seq, f"groundtruth_{N}.txt")

        if os.path.exists(dst_rgb) and os.path.exists(dst_gt):
            continue  # already exists

        actual_n = min(N, n_total)
        selected_frames = all_frames[:actual_n]
        selected_poses = all_poses[:actual_n]

        os.makedirs(dst_rgb, exist_ok=True)
        for fname in selected_frames:
            src = os.path.join(os.path.abspath(src_rgb), fname)
            dst = os.path.join(dst_rgb, fname)
            if not os.path.exists(dst):
                os.symlink(src, dst)

        with open(dst_gt, "w") as f:
            f.writelines(selected_poses)

        print(f"  {seq}: rgb_{N} -> {actual_n} frames (symlinked)")

print("\nDone.")
