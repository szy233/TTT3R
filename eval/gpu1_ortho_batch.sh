#!/bin/bash
# GPU1 batch: ortho hyperparam sensitivity (TUM) + video depth + 7scenes
# All serial on GPU1

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=src

LOG=eval/gpu1_ortho_batch.log
exec > >(tee -a "$LOG") 2>&1

echo "[$(date)] ========== GPU1 Ortho Batch Start =========="

# ============================================================
# Part 1: Ortho Hyperparameter Sensitivity — TUM relpose
# Default: α_novel=0.5, α_drift=0.05, β=0.95
# Varying one param at a time
# ============================================================

echo "[$(date)] === Part 1: Ortho Hyperparam Sensitivity (TUM) ==="

# α_novel sweep (fix α_drift=0.05, β=0.95)
for AN in 0.3 0.7; do
    NAME="ttt3r_ortho_an${AN}_ad005"
    echo "[$(date)] START: $NAME"
    accelerate launch --num_processes 1 --main_process_port 29561 \
        eval/relpose/launch.py \
        --weights model/cut3r_512_dpt_4_64.pth \
        --output_dir eval_results/relpose/tum_s1_1000/$NAME \
        --eval_dataset tum_s1_1000 --size 512 --model_update_type ttt3r_ortho \
        --ortho_alpha_novel $AN --ortho_alpha_drift 0.05 --ortho_beta 0.95
    echo "[$(date)] DONE: $NAME"
done

# α_drift sweep (fix α_novel=0.5, β=0.95)
for AD in 0.1 0.2; do
    NAME="ttt3r_ortho_an05_ad${AD}"
    echo "[$(date)] START: $NAME"
    accelerate launch --num_processes 1 --main_process_port 29561 \
        eval/relpose/launch.py \
        --weights model/cut3r_512_dpt_4_64.pth \
        --output_dir eval_results/relpose/tum_s1_1000/$NAME \
        --eval_dataset tum_s1_1000 --size 512 --model_update_type ttt3r_ortho \
        --ortho_alpha_novel 0.5 --ortho_alpha_drift $AD --ortho_beta 0.95
    echo "[$(date)] DONE: $NAME"
done

# β sweep (fix α_novel=0.5, α_drift=0.05)
for BETA in 0.9 0.99; do
    NAME="ttt3r_ortho_an05_ad005_b${BETA}"
    echo "[$(date)] START: $NAME"
    accelerate launch --num_processes 1 --main_process_port 29561 \
        eval/relpose/launch.py \
        --weights model/cut3r_512_dpt_4_64.pth \
        --output_dir eval_results/relpose/tum_s1_1000/$NAME \
        --eval_dataset tum_s1_1000 --size 512 --model_update_type ttt3r_ortho \
        --ortho_alpha_novel 0.5 --ortho_alpha_drift 0.05 --ortho_beta $BETA
    echo "[$(date)] DONE: $NAME"
done

echo "[$(date)] === Part 1 Complete ==="

# ============================================================
# Part 2: Video Depth — ttt3r_ortho on KITTI, Bonn, Sintel
# ============================================================

echo "[$(date)] === Part 2: Video Depth (ttt3r_ortho) ==="

for DATASET in kitti_s1_500 bonn_s1_500 sintel; do
    echo "[$(date)] START: video_depth $DATASET ttt3r_ortho"
    accelerate launch --num_processes 1 --main_process_port 29561 \
        eval/video_depth/launch.py \
        --weights model/cut3r_512_dpt_4_64.pth \
        --output_dir eval_results/video_depth/$DATASET/ttt3r_ortho \
        --eval_dataset $DATASET --size 512 --model_update_type ttt3r_ortho \
        --ortho_alpha_novel 0.5 --ortho_alpha_drift 0.05 --ortho_beta 0.95
    echo "[$(date)] DONE: video_depth $DATASET ttt3r_ortho"
done

echo "[$(date)] === Part 2 Complete ==="

# ============================================================
# Part 3: 7scenes 3D Reconstruction — ttt3r_ortho
# ============================================================

echo "[$(date)] === Part 3: 7scenes Recon (ttt3r_ortho) ==="

echo "[$(date)] START: 7scenes ttt3r_ortho"
accelerate launch --num_processes 1 --main_process_port 29561 \
    eval/mv_recon/launch.py \
    --weights model/cut3r_512_dpt_4_64.pth \
    --output_dir eval_results/video_recon/7scenes_200/ttt3r_ortho \
    --size 512 --max_frames 200 --model_update_type ttt3r_ortho \
    --ortho_alpha_novel 0.5 --ortho_alpha_drift 0.05 --ortho_beta 0.95
echo "[$(date)] DONE: 7scenes ttt3r_ortho"

echo "[$(date)] === Part 3 Complete ==="

echo "[$(date)] ========== GPU1 Ortho Batch ALL DONE =========="
