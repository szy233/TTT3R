#!/bin/bash
# Wait for the diagnosis batch, then sweep the dampening coefficient on Point3R/TUM.
cd /home/szy/research/TTT3R/.claude/worktrees/understand-full-project-14f204
LOG=rebuttal/results/point3r_alpha.log
while pgrep -u szy -f point3r_diagnose >/dev/null; do sleep 60; done
echo "=== diagnosis batch finished, starting alpha sweep $(date +%H:%M) ===" >> $LOG
for a in 1.0 0.5 0.33; do
  echo ">>> $(date +%H:%M) alpha=$a" >> $LOG
  bash rebuttal/scripts/run_point3r_eval.sh 1 $a tum >> $LOG 2>&1 || echo "  FAILED alpha=$a" >> $LOG
done
echo "=== ALPHA_SWEEP_DONE $(date +%H:%M) ===" >> $LOG
