#!/bin/bash
# Rebuild the kitti_odo_full layout used by eval/relpose/kitti_odo_full_report.md.
# Everything is a symlink into the raw KITTI tree plus TUM-format pose files, so
# the images are not copied. Run from the repository root.
set -e

RAW=/mnt/sda/rkj/VGGT_Long_outdoor/kitti_00_10
OUT=rebuttal/iclr2027/data
PY=${DDD3R_PYTHON:-/home/szy/anaconda3/envs/ttt3r/bin/python}
SEQS="00 01 02 03 04 05 06 07 08 09 10"

mkdir -p "$OUT/kitti_src/sequences" "$OUT/kitti_src/poses"
for seq in $SEQS; do
    ln -sfn "$RAW/$seq" "$OUT/kitti_src/sequences/$seq"
    ln -sfn "$RAW/gt_poses_kitti/$seq.txt" "$OUT/kitti_src/poses/$seq.txt"
done

# --lengths 99999 skips the fixed-length tags, leaving only image_full and pose_full.
"$PY" rebuttal/iclr2027/scripts/prepare_kitti_odometry.py \
    --kitti_root "$OUT/kitti_src" \
    --output_root "$OUT/long_kitti_odo_s1" \
    --seqs $SEQS --lengths 99999 --full
