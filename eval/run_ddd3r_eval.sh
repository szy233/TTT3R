#!/bin/bash
# =============================================================================
# DDD3R Unified Evaluation Script (paper naming)
# Usage: bash eval/run_ddd3r_eval.sh [GPU_ID] [DATASET] [METHOD]
#
# Methods: cut3r, ttt3r, ddd3r_constant, ddd3r_brake, ddd3r, ddd3r_g{N}
# Datasets: tum_s1_1000, scannet_s3_1000, sintel, kitti, bonn, 7scenes
# =============================================================================

set -e

GPU=${1:-0}
DATASET=${2:-tum_s1_1000}
METHOD=${3:-ddd3r}

export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=src
PY=/home/szy/anaconda3/envs/ttt3r/bin/python
WEIGHTS="model/cut3r_512_dpt_4_64.pth"
PORT=$((29560 + GPU))

# Parse method → model_update_type + DDD3R params
case "$METHOD" in
    cut3r)
        UPDATE_TYPE="cut3r"
        EXTRA_ARGS=""
        ;;
    ttt3r)
        UPDATE_TYPE="ttt3r"
        EXTRA_ARGS=""
        ;;
    ddd3r_constant)
        UPDATE_TYPE="ddd3r_constant"
        EXTRA_ARGS="--alpha 0.5"
        ;;
    ddd3r_constant_p*)
        # e.g. ddd3r_constant_p033 → alpha=0.33
        P=$(echo "$METHOD" | sed 's/ddd3r_constant_p/0./')
        UPDATE_TYPE="ddd3r_constant"
        EXTRA_ARGS="--alpha $P"
        ;;
    ddd3r_brake)
        UPDATE_TYPE="ddd3r_brake"
        EXTRA_ARGS="--brake_tau 2.0"
        ;;
    ddd3r)
        UPDATE_TYPE="ddd3r"
        EXTRA_ARGS="--alpha_perp 0.5 --alpha_parallel 0.05 --beta_ema 0.95 --gamma 0"
        ;;
    ddd3r_g*)
        # e.g. ddd3r_g2 → gamma=2, ddd3r_g0.5 → gamma=0.5
        GAMMA=$(echo "$METHOD" | sed 's/ddd3r_g//')
        UPDATE_TYPE="ddd3r"
        EXTRA_ARGS="--alpha_perp 0.5 --alpha_parallel 0.05 --beta_ema 0.95 --gamma $GAMMA"
        METHOD="ddd3r_g${GAMMA}"
        ;;
    *)
        echo "Unknown method: $METHOD"
        echo "Available: cut3r, ttt3r, ddd3r_constant, ddd3r_brake, ddd3r, ddd3r_g{N}"
        exit 1
        ;;
esac

# Parse dataset → eval task type
case "$DATASET" in
    tum_*|scannet_*|sintel)
        TASK="relpose"
        LAUNCH="eval/relpose/launch.py"
        ;;
    kitti|bonn|sintel_depth)
        TASK="video_depth"
        LAUNCH="eval/video_depth/launch.py"
        ;;
    7scenes)
        TASK="mv_recon"
        LAUNCH="eval/mv_recon/launch.py"
        ;;
    *)
        echo "Unknown dataset: $DATASET"
        exit 1
        ;;
esac

OUTPUT_DIR="eval_results/${TASK}/${DATASET}/${METHOD}"

echo "=== DDD3R Eval: ${METHOD} on ${DATASET} (GPU ${GPU}) ==="
echo "  update_type: ${UPDATE_TYPE}"
echo "  output_dir:  ${OUTPUT_DIR}"

$PY -m accelerate.commands.launch --num_processes 1 --main_process_port $PORT \
    $LAUNCH \
    --weights $WEIGHTS --size 512 \
    --output_dir $OUTPUT_DIR \
    --eval_dataset $DATASET \
    --model_update_type $UPDATE_TYPE \
    $EXTRA_ARGS

echo "=== Done: ${METHOD} on ${DATASET} ==="
