#!/bin/bash
# Length-aware ortho: sweep warmup T0 on TUM 1000f + ScanNet 90f
# Usage: CUDA_VISIBLE_DEVICES=0 bash eval/run_warmup_ortho.sh

set -e

WEIGHTS="model/cut3r_512_dpt_4_64.pth"
BASE_ARGS="--size 512 --ortho_alpha_novel 0.5 --ortho_alpha_drift 0.05 --ortho_beta 0.95"
PORT=29570

# T0 values to sweep; window = T0 (linear ramp over same length)
T0_VALUES=(10 20 50)

echo "=== TUM 1000f — warmup sweep ==="
for T0 in "${T0_VALUES[@]}"; do
    W=$T0
    TAG="ttt3r_ortho_warmup_t${T0}_w${W}"
    OUT="eval_results/relpose/tum_s1_1000/${TAG}"
    echo "--- T0=${T0}, window=${W} → ${OUT}"
    PYTHONPATH=src accelerate launch --num_processes 1 --main_process_port $PORT \
        eval/relpose/launch.py \
        --weights $WEIGHTS --output_dir $OUT \
        --eval_dataset tum_s1_1000 $BASE_ARGS \
        --model_update_type ttt3r_ortho \
        --ortho_warmup_t0 $T0 --ortho_warmup_window $W
done

echo "=== ScanNet 90f (first-90) — warmup sweep ==="
for T0 in "${T0_VALUES[@]}"; do
    W=$T0
    TAG="ttt3r_ortho_warmup_t${T0}_w${W}"
    OUT="eval_results/relpose/scannet_s3_90_first/${TAG}"
    echo "--- T0=${T0}, window=${W} → ${OUT}"
    PYTHONPATH=src accelerate launch --num_processes 1 --main_process_port $PORT \
        eval/relpose/launch.py \
        --weights $WEIGHTS --output_dir $OUT \
        --eval_dataset scannet_s3_90 $BASE_ARGS \
        --model_update_type ttt3r_ortho \
        --ortho_warmup_t0 $T0 --ortho_warmup_window $W
done

echo "=== ScanNet 1000f — warmup sweep ==="
for T0 in "${T0_VALUES[@]}"; do
    W=$T0
    TAG="ttt3r_ortho_warmup_t${T0}_w${W}"
    OUT="eval_results/relpose/scannet_s3_1000/${TAG}"
    echo "--- T0=${T0}, window=${W} → ${OUT}"
    PYTHONPATH=src accelerate launch --num_processes 1 --main_process_port $PORT \
        eval/relpose/launch.py \
        --weights $WEIGHTS --output_dir $OUT \
        --eval_dataset scannet_s3_1000 $BASE_ARGS \
        --model_update_type ttt3r_ortho \
        --ortho_warmup_t0 $T0 --ortho_warmup_window $W
done

echo "Done. Check eval_results/relpose/*/ttt3r_ortho_warmup_*/_error_log.txt"
