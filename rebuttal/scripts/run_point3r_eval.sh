#!/bin/bash
# =============================================================================
# E5 step 2 — Does the M1 fix (constant dampening) transfer to Point3R?
#
# Point3R's memory merge fully replaces a pointer feature (beta=1). The DDD3R
# hook in src/dust3r/point3r.py turns that into  mem <- mem + alpha*(new - mem).
# DDD3R_ALPHA=1.0 reproduces upstream exactly, so it doubles as our baseline.
#
# Usage: bash rebuttal/scripts/run_point3r_eval.sh <GPU> <ALPHA> [DATASET]
# =============================================================================
set -e
GPU=${1:-1}; ALPHA=${2:-1.0}; DS=${3:-tum}
ROOT=/home/szy/research/TTT3R/.claude/worktrees/understand-full-project-14f204
P3R=$ROOT/rebuttal/external/Point3R
TAG=$(echo "$ALPHA" | tr -d '.')
OUT=$ROOT/rebuttal/results/point3r/${DS}_alpha${TAG}
mkdir -p "$OUT"
cd $P3R
export CUDA_VISIBLE_DEVICES=$GPU
export DDD3R_ALPHA=$ALPHA
export PYTHONPATH=$P3R:$P3R/src
echo "=== Point3R eval: $DS alpha=$ALPHA -> $OUT ==="
python3 -m accelerate.commands.launch --num_processes 1 --main_process_port $((29900+GPU)) \
  eval/relpose/launch.py \
  --weights $ROOT/rebuttal/external/Point3R_ckpt/point3r.pth \
  --output_dir "$OUT" --eval_dataset $DS --size 512
echo "=== done alpha=$ALPHA ==="
