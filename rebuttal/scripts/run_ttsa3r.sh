#!/bin/bash
# =============================================================================
# REBUTTAL EXPERIMENT E1 — TTSA3R (TAUM x SCUM) head-to-head baseline
#
# Addresses Reviewer 1ake (R3-C2): "TTSA3R / TAUM is discussed analytically (M2)
# but is not given a head-to-head row ... leaving the prediction
# 'TAUM == constant dampening' empirically unconfirmed."
#
# Runs the TTSA3R gate as a live update type (src/dust3r/model.py, update_type
# "ttsa3r") on the long-sequence relpose benchmarks, so it can be placed in the
# same table as CUT3R / TTT3R / DDD3R.
#
# Results are written under rebuttal/results/ to stay separate from the main
# eval_results/ tree.
#
# Usage: bash rebuttal/scripts/run_ttsa3r.sh <GPU> <DATASET>
#   e.g. bash rebuttal/scripts/run_ttsa3r.sh 1 tum_s1_1000
# =============================================================================
set -e

GPU=${1:-1}
DATASET=${2:-tum_s1_1000}
METHOD=ttsa3r

export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=src
PY=${DDD3R_PYTHON:-/home/szy/anaconda3/envs/ttt3r/bin/python}
WEIGHTS="model/cut3r_512_dpt_4_64.pth"
PORT=$((29700 + GPU))

case "$DATASET" in
    tum|tum_*|scannet_*|sintel|kitti_odom)
        TASK="relpose";    LAUNCH="eval/relpose/launch.py" ;;
    kitti|kitti_s1_*|bonn|bonn_s1_*|sintel_depth)
        TASK="video_depth"; LAUNCH="eval/video_depth/launch.py" ;;
    *) echo "Unknown dataset: $DATASET"; exit 1 ;;
esac

EVAL_DATASET="$DATASET"
[ "$DATASET" = "sintel_depth" ] && EVAL_DATASET="sintel"

OUTPUT_DIR="rebuttal/results/${TASK}/${DATASET}/${METHOD}"
mkdir -p "$OUTPUT_DIR"

echo "=== REBUTTAL E1: ${METHOD} on ${DATASET} (GPU ${GPU}) ==="
echo "  output_dir: ${OUTPUT_DIR}"

$PY -m accelerate.commands.launch --num_processes 1 --main_process_port $PORT \
    $LAUNCH \
    --weights $WEIGHTS --size 512 \
    --output_dir "$OUTPUT_DIR" \
    --eval_dataset $EVAL_DATASET \
    --model_update_type ttsa3r

if [ "$TASK" = "video_depth" ]; then
    $PY eval/video_depth/eval_depth.py \
        --output_dir "$OUTPUT_DIR" \
        --eval_dataset $EVAL_DATASET \
        --align "scale&shift"
fi

echo "=== Done: ${METHOD} on ${DATASET} ==="
