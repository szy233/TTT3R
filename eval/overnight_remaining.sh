#!/bin/bash
# 修复后继续跑剩余的过夜实验
# Sintel 数据集名应为 sintel（不是 sintel_s1_500）

set -e
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=src

WEIGHTS="model/cut3r_512_dpt_4_64.pth"
PORT=29570

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# ============================================================
# 3c. momentum_inv_t1 — Sintel video depth (fix: sintel not sintel_s1_500)
# ============================================================
log "START: momentum_inv_t1 Sintel video depth"
python -m accelerate.commands.launch --num_processes 1 --main_process_port $PORT \
    eval/video_depth/launch.py \
    --weights $WEIGHTS \
    --output_dir eval_results/video_depth/sintel/ttt3r_momentum_inv_t1 \
    --eval_dataset sintel --size 512 \
    --model_update_type ttt3r_momentum \
    --momentum_tau 1.0
log "DONE: momentum_inv_t1 Sintel"

# ============================================================
# 4. brake_geo — Video Depth (KITTI + Bonn + Sintel)
# ============================================================
log "START: brake_geo KITTI video depth"
python -m accelerate.commands.launch --num_processes 1 --main_process_port $PORT \
    eval/video_depth/launch.py \
    --weights $WEIGHTS \
    --output_dir eval_results/video_depth/kitti_s1_500/ttt3r_brake_geo \
    --eval_dataset kitti_s1_500 --size 512 \
    --model_update_type ttt3r_brake_geo \
    --momentum_tau 1.0 \
    --geo_gate_tau 2.0 --geo_gate_freq_cutoff 4
log "DONE: brake_geo KITTI"

log "START: brake_geo Bonn video depth"
python -m accelerate.commands.launch --num_processes 1 --main_process_port $PORT \
    eval/video_depth/launch.py \
    --weights $WEIGHTS \
    --output_dir eval_results/video_depth/bonn_s1_500/ttt3r_brake_geo \
    --eval_dataset bonn_s1_500 --size 512 \
    --model_update_type ttt3r_brake_geo \
    --momentum_tau 1.0 \
    --geo_gate_tau 2.0 --geo_gate_freq_cutoff 4
log "DONE: brake_geo Bonn"

log "START: brake_geo Sintel video depth"
python -m accelerate.commands.launch --num_processes 1 --main_process_port $PORT \
    eval/video_depth/launch.py \
    --weights $WEIGHTS \
    --output_dir eval_results/video_depth/sintel/ttt3r_brake_geo \
    --eval_dataset sintel --size 512 \
    --model_update_type ttt3r_brake_geo \
    --momentum_tau 1.0 \
    --geo_gate_tau 2.0 --geo_gate_freq_cutoff 4
log "DONE: brake_geo Sintel"

# ============================================================
# 5. momentum_inv_t1 — 7scenes 3D Recon
# ============================================================
log "START: momentum_inv_t1 7scenes recon"
python -m accelerate.commands.launch --num_processes 1 --main_process_port $PORT \
    eval/mv_recon/launch.py \
    --weights $WEIGHTS \
    --output_dir eval_results/video_recon/7scenes_200/ttt3r_momentum_inv_t1 \
    --size 512 --max_frames 200 \
    --model_update_type ttt3r_momentum \
    --momentum_tau 1.0
log "DONE: momentum_inv_t1 7scenes"

# ============================================================
# 6. brake_geo — 7scenes 3D Recon
# ============================================================
log "START: brake_geo 7scenes recon"
python -m accelerate.commands.launch --num_processes 1 --main_process_port $PORT \
    eval/mv_recon/launch.py \
    --weights $WEIGHTS \
    --output_dir eval_results/video_recon/7scenes_200/ttt3r_brake_geo \
    --size 512 --max_frames 200 \
    --model_update_type ttt3r_brake_geo \
    --momentum_tau 1.0 \
    --geo_gate_tau 2.0 --geo_gate_freq_cutoff 4
log "DONE: brake_geo 7scenes"

log "=== ALL REMAINING EXPERIMENTS COMPLETE ==="
