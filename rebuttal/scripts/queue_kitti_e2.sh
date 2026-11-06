#!/bin/bash
# Queue: KITTI baselines for the TTSA3R row (E1 completion), then E2 in-the-wild demo.
# Sequential on one GPU so we do not contend with other users' jobs on this machine.
cd /home/szy/research/TTT3R/.claude/worktrees/understand-full-project-14f204
GPU=${1:-1}
LOG=rebuttal/results/kitti_queue.log
echo "=== queue start $(date +%H:%M) ===" >> $LOG
for m in cut3r ttt3r ddd3r_constant ddd3r_brake ddd3r; do
  if [ -d "rebuttal/results/relpose/kitti_odom/$m" ] && \
     [ "$(ls rebuttal/results/relpose/kitti_odom/$m/*_eval_metric.txt 2>/dev/null | wc -l)" -ge 11 ]; then
    echo ">>> skip $m (already 11 seqs)" >> $LOG; continue
  fi
  echo ">>> $(date +%H:%M) kitti_odom / $m" >> $LOG
  bash rebuttal/scripts/run_arkit_eval.sh $GPU kitti_odom $m >> $LOG 2>&1 || echo "  (ended: $m)" >> $LOG
done
echo "=== KITTI_DONE $(date +%H:%M), starting E2 ===" >> $LOG
bash rebuttal/scripts/run_inwild_demo.sh $GPU >> $LOG 2>&1
echo "=== ALL_DONE $(date +%H:%M) ===" >> $LOG
