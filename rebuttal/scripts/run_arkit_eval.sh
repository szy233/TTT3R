#!/bin/bash
# =============================================================================
# REBUTTAL EXPERIMENT E4 — ARKitScenes relpose eval
#
# Addresses Reviewer ULz9 (R1-C1): real-world / in-the-wild generalization.
# Tests the C3 drift-energy prediction registered in
# rebuttal/docs/E4_arkitscenes_PREREGISTRATION.md.
#
# Results go under rebuttal/results/ to stay out of the main eval_results/ tree.
#
# Usage: bash rebuttal/scripts/run_arkit_eval.sh <GPU> <DATASET> <METHOD>
#   e.g. bash rebuttal/scripts/run_arkit_eval.sh 1 arkit_s1_1000 ddd3r
# =============================================================================
set -e

GPU=${1:-1}
DATASET=${2:-arkit_s1_1000}
METHOD=${3:-cut3r}

export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=src
PY=${DDD3R_PYTHON:-/home/szy/anaconda3/envs/ttt3r/bin/python}
WEIGHTS="model/cut3r_512_dpt_4_64.pth"
PORT=$((29800 + GPU))

case "$METHOD" in
    cut3r)          UPDATE_TYPE="cut3r";          EXTRA_ARGS="" ;;
    ttt3r)          UPDATE_TYPE="ttt3r";          EXTRA_ARGS="" ;;
    ttsa3r)         UPDATE_TYPE="ttsa3r";         EXTRA_ARGS="" ;;
    ddd3r_constant) UPDATE_TYPE="ddd3r_constant"; EXTRA_ARGS="--alpha 0.5" ;;
    ddd3r_brake)    UPDATE_TYPE="ddd3r_brake";    EXTRA_ARGS="--brake_tau 1.0" ;;
    ddd3r)          UPDATE_TYPE="ddd3r";
                    EXTRA_ARGS="--alpha_perp 0.5 --alpha_parallel 0.05 --beta_ema 0.95 --gamma 0" ;;
    *) echo "Unknown method: $METHOD"; exit 1 ;;
esac

OUTPUT_DIR="rebuttal/results/relpose/${DATASET}/${METHOD}"
mkdir -p "$OUTPUT_DIR"

echo "=== E4: ${METHOD} on ${DATASET} (GPU ${GPU}) ==="

$PY -m accelerate.commands.launch --num_processes 1 --main_process_port $PORT \
    eval/relpose/launch.py \
    --weights $WEIGHTS --size 512 \
    --output_dir "$OUTPUT_DIR" \
    --eval_dataset $DATASET \
    --model_update_type $UPDATE_TYPE \
    $EXTRA_ARGS

echo "=== Done: ${METHOD} on ${DATASET} ==="
