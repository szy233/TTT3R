#!/bin/bash
# 过夜实验脚本 — GPU0 串行
# 2026-03-27 晚, 预估总时长 ~8h
# 当前 l2gate_fixed ScanNet 在跑 (PID 3096412)，等它完成后手动启动本脚本
# 或者直接启动，前面的实验不冲突就行

set -e
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=src

WEIGHTS="model/cut3r_512_dpt_4_64.pth"
PORT=29570

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# ============================================================
# 1. ttt3r_brake_geo — ScanNet relpose (~4h)
# ============================================================
log "START: brake_geo ScanNet relpose"
python -m accelerate.commands.launch --num_processes 1 --main_process_port $PORT \
    eval/relpose/launch.py \
    --weights $WEIGHTS \
    --output_dir eval_results/relpose/scannet_s3_1000/ttt3r_brake_geo \
    --eval_dataset scannet_s3_1000 --size 512 \
    --model_update_type ttt3r_brake_geo \
    --momentum_tau 1.0 \
    --geo_gate_tau 2.0 --geo_gate_freq_cutoff 4
log "DONE: brake_geo ScanNet relpose"

# ============================================================
# 2. ttt3r_brake_geo — TUM relpose (~30min)
# ============================================================
log "START: brake_geo TUM relpose"
python -m accelerate.commands.launch --num_processes 1 --main_process_port $PORT \
    eval/relpose/launch.py \
    --weights $WEIGHTS \
    --output_dir eval_results/relpose/tum_s1_1000/ttt3r_brake_geo \
    --eval_dataset tum_s1_1000 --size 512 \
    --model_update_type ttt3r_brake_geo \
    --momentum_tau 1.0 \
    --geo_gate_tau 2.0 --geo_gate_freq_cutoff 4
log "DONE: brake_geo TUM relpose"

# ============================================================
# 3. ttt3r_momentum_inv_t1 — Video Depth (KITTI + Bonn + Sintel, ~1h)
# ============================================================
log "START: momentum_inv_t1 KITTI video depth"
python -m accelerate.commands.launch --num_processes 1 --main_process_port $PORT \
    eval/video_depth/launch.py \
    --weights $WEIGHTS \
    --output_dir eval_results/video_depth/kitti_s1_500/ttt3r_momentum_inv_t1 \
    --eval_dataset kitti_s1_500 --size 512 \
    --model_update_type ttt3r_momentum \
    --momentum_tau 1.0
log "DONE: momentum_inv_t1 KITTI"

log "START: momentum_inv_t1 Bonn video depth"
python -m accelerate.commands.launch --num_processes 1 --main_process_port $PORT \
    eval/video_depth/launch.py \
    --weights $WEIGHTS \
    --output_dir eval_results/video_depth/bonn_s1_500/ttt3r_momentum_inv_t1 \
    --eval_dataset bonn_s1_500 --size 512 \
    --model_update_type ttt3r_momentum \
    --momentum_tau 1.0
log "DONE: momentum_inv_t1 Bonn"

log "START: momentum_inv_t1 Sintel video depth"
python -m accelerate.commands.launch --num_processes 1 --main_process_port $PORT \
    eval/video_depth/launch.py \
    --weights $WEIGHTS \
    --output_dir eval_results/video_depth/sintel_s1_500/ttt3r_momentum_inv_t1 \
    --eval_dataset sintel_s1_500 --size 512 \
    --model_update_type ttt3r_momentum \
    --momentum_tau 1.0
log "DONE: momentum_inv_t1 Sintel"

# ============================================================
# 4. ttt3r_brake_geo — Video Depth (KITTI + Bonn + Sintel, ~1h)
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
    --output_dir eval_results/video_depth/sintel_s1_500/ttt3r_brake_geo \
    --eval_dataset sintel_s1_500 --size 512 \
    --model_update_type ttt3r_brake_geo \
    --momentum_tau 1.0 \
    --geo_gate_tau 2.0 --geo_gate_freq_cutoff 4
log "DONE: brake_geo Sintel"

# ============================================================
# 5. ttt3r_momentum_inv_t1 — 7scenes 3D Recon (~1.5h)
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
# 6. ttt3r_brake_geo — 7scenes 3D Recon (~1.5h)
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

log "=== ALL OVERNIGHT EXPERIMENTS COMPLETE ==="
