#!/bin/bash
# =============================================================================
# E4 step 3 — ARKitScenes ATE across all methods.
#
# DO NOT RUN until the P2 branch has been locked into
# rebuttal/docs/E4_arkitscenes_PREREGISTRATION.md and committed.
# The pre-registration requires drift energy to be measured and the predicted
# ranking recorded BEFORE any ATE exists.
#
# Usage: bash rebuttal/scripts/run_arkit_all.sh <GPU>
# =============================================================================
set -e
GPU=${1:-1}
DS=arkit_s1_500
LOG=rebuttal/results/arkit_ate.log

cd /home/szy/research/TTT3R/.claude/worktrees/understand-full-project-14f204

if ! /bin/grep -q "P2 BRANCH LOCKED" rebuttal/docs/E4_arkitscenes_PREREGISTRATION.md; then
  echo "REFUSING TO RUN: P2 branch not yet locked in the pre-registration." >&2
  echo "Measure drift energy, record the predicted ranking, commit, then rerun." >&2
  exit 1
fi

echo "=== ARKitScenes ATE queue started $(date +%H:%M) ===" >> $LOG
for m in cut3r ttt3r ttsa3r ddd3r_constant ddd3r_brake ddd3r; do
  echo ">>> $(date +%H:%M) $m on $DS" >> $LOG
  bash rebuttal/scripts/run_arkit_eval.sh "$GPU" "$DS" "$m" >> $LOG 2>&1 \
    || echo "  FAILED $m" >> $LOG
done
echo "=== ARKIT_ATE_ALLDONE ===" >> $LOG
