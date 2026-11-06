#!/bin/bash
# =============================================================================
# REBUTTAL EXPERIMENT E2 — in-the-wild qualitative demo
#
# Addresses Reviewer ULz9 (R1-C1 / Q1): behaviour on casual, uncalibrated video.
# These clips have NO ground-truth trajectory, so this is qualitative ONLY —
# no quantitative claim may be derived from it.
#
# demo.py saves reconstruction output via prepare_output() *before* it launches
# the blocking viser viewer, so we run it and stop once the viewer line appears.
#
# Usage: bash rebuttal/scripts/run_inwild_demo.sh <GPU>
# =============================================================================
set -u
GPU=${1:-1}
cd /home/szy/research/TTT3R/.claude/worktrees/understand-full-project-14f204
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=src
PY=${DDD3R_PYTHON:-/home/szy/anaconda3/envs/ttt3r/bin/python}
LOG=rebuttal/results/inwild.log

for clip in westlake taylor; do
  for m in cut3r ttt3r ddd3r_constant ddd3r_brake ddd3r; do
    OUT="rebuttal/results/inwild/${clip}/${m}"
    if [ "$(/bin/ls "$OUT" 2>/dev/null | wc -l)" -ge 4 ]; then
      echo ">>> skip $clip/$m (already done)" >> $LOG; continue
    fi
    mkdir -p "$OUT"
    echo ">>> $(date +%H:%M) $clip / $m" >> $LOG
    RLOG=rebuttal/results/inwild_${clip}_${m}.log
    $PY demo.py \
      --model_path model/cut3r_512_dpt_4_64.pth --size 512 \
      --seq_path examples/${clip}.mp4 --output_dir "$OUT" \
      --model_update_type $m --frame_interval 3 --reset_interval 1000 \
      --downsample_factor 100 --vis_threshold 6.0 --port $((9100 + GPU)) \
      > $RLOG 2>&1 &
    pid=$!
    # stop as soon as outputs are on disk (viewer line) or after a hard cap
    for _ in $(seq 1 900); do
      sleep 4
      if /bin/grep -q "Launching point cloud viewer" $RLOG; then break; fi
      if ! kill -0 $pid 2>/dev/null; then break; fi
    done
    pkill -P $pid 2>/dev/null
    kill -9 $pid 2>/dev/null
    sleep 3
    echo "    saved: $(/bin/ls $OUT 2>/dev/null | wc -l) files (log: $RLOG)" >> $LOG
  done
done
echo "=== INWILD_ALLDONE ===" >> $LOG
