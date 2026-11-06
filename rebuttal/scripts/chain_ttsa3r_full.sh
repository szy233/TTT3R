#!/bin/bash
# Fill the complete TTSA3R row across both main tables (Reviewer 1ake R3-C2).
# Runs sequentially on GPU1 after the in-progress ScanNet 1000f run finishes.
cd /home/szy/research/TTT3R/.claude/worktrees/understand-full-project-14f204
LOG=rebuttal/results/ttsa3r_full.log
echo "=== queue started ===" >> $LOG
# wait for the ScanNet 1000f run to finish
while pgrep -f "scannet_s3_1000 --model_update_type ttsa3r" >/dev/null; do sleep 120; done
echo "scannet_s3_1000 done, starting remaining cells" >> $LOG

# order: cheap+important first
for spec in \
    "tum:relpose" \
    "scannet_s3_90:relpose" \
    "bonn_s1_500:vdepth" \
    "sintel_depth:vdepth" \
    "kitti_s1_500:vdepth" \
    "kitti_odom:relpose" ; do
  ds="${spec%%:*}"
  echo ">>> $(date +%H:%M) running ttsa3r on $ds" >> $LOG
  bash rebuttal/scripts/run_ttsa3r.sh 1 "$ds" >> $LOG 2>&1 || echo "  FAILED $ds" >> $LOG
done
echo "=== TTSA3R_FULL_ALLDONE ===" >> $LOG
