#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=src
LOG=eval/gpu1_adaptive_threshold.log
exec > >(tee -a "$LOG") 2>&1
echo "[$(date)] ========== Adaptive Threshold Start =========="

for DATASET in tum_s1_1000 scannet_s3_1000; do
    echo "[$(date)] START: $DATASET threshold"
    accelerate launch --num_processes 1 --main_process_port 29561 \
        eval/relpose/launch.py \
        --weights model/cut3r_512_dpt_4_64.pth \
        --output_dir eval_results/relpose/$DATASET/ttt3r_ortho_threshold \
        --eval_dataset $DATASET --size 512 --model_update_type ttt3r_ortho \
        --ortho_alpha_novel 0.5 --ortho_alpha_drift 0.05 --ortho_beta 0.95 \
        --ortho_adaptive threshold
    echo "[$(date)] DONE: $DATASET threshold"
done

echo "[$(date)] ========== ALL DONE =========="
