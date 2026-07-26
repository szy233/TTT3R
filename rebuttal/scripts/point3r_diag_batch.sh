#!/bin/bash
# E5 step 1b — confirm the Point3R diagnosis is stable across sequences and datasets.
cd /home/szy/research/TTT3R/.claude/worktrees/understand-full-project-14f204
GPU=${1:-1}
CKPT=rebuttal/external/Point3R_ckpt/point3r.pth
LOG=rebuttal/results/point3r_diag_batch.log
: > $LOG
run () {  # name, dir
  echo ">>> $1" >> $LOG
  CUDA_VISIBLE_DEVICES=$GPU python3 rebuttal/scripts/point3r_diagnose.py \
    --ckpt $CKPT --seq_dir "$2" --n_frames 100 \
    --out rebuttal/results/point3r_diag_$1.json 2>&1 \
    | /bin/grep -E "M1 relative|M3 cos|M3 drift|memory grows|overwrite events" >> $LOG
}
for s in walking_xyz walking_static sitting_xyz sitting_halfsphere; do
  run "tum_$s" "data/long_tum_s1/rgbd_dataset_freiburg3_$s/rgb_1000"
done
for d in $(/bin/ls -d data/long_scannet_s3/*/ 2>/dev/null | head -3); do
  n=$(basename $d)
  [ -d "$d/color_1000" ] && run "scannet_$n" "$d/color_1000"
done
echo "=== DIAG_BATCH_DONE ===" >> $LOG
