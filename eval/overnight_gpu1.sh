#!/bin/bash
# GPU1 overnight experiments: ScanNet 90f eval (5 configs) + Sintel relpose
# 2026-03-28 晚
set -e
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=src
export PYTHONUNBUFFERED=1

WEIGHTS="model/cut3r_512_dpt_4_64.pth"
PORT=29571
LOG="eval_results/overnight_gpu1.log"
mkdir -p eval_results

log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG"; }

log "=== GPU1 overnight started ==="

# ============================================================
# 1. ScanNet Short Sequence Eval — 90 frames (5 configs, ~5x20min = 1.5h)
#    Matching TTSA3R Table 2 protocol
# ============================================================
COMMON_90="--weights $WEIGHTS --eval_dataset scannet_s3_90 --size 512"

# cut3r baseline
log "START: ScanNet90 cut3r"
accelerate launch --num_processes 1 --main_process_port $PORT \
    eval/relpose/launch.py $COMMON_90 \
    --model_update_type cut3r \
    --output_dir eval_results/relpose/scannet_s3_90/cut3r \
    2>&1 | tee -a "$LOG"
log "DONE: ScanNet90 cut3r"

# ttt3r
log "START: ScanNet90 ttt3r"
accelerate launch --num_processes 1 --main_process_port $PORT \
    eval/relpose/launch.py $COMMON_90 \
    --model_update_type ttt3r \
    --output_dir eval_results/relpose/scannet_s3_90/ttt3r \
    2>&1 | tee -a "$LOG"
log "DONE: ScanNet90 ttt3r"

# ttt3r_random p=0.5 (ScanNet optimal)
log "START: ScanNet90 ttt3r_random p=0.5"
accelerate launch --num_processes 1 --main_process_port $PORT \
    eval/relpose/launch.py $COMMON_90 \
    --model_update_type ttt3r_random --random_gate_p 0.5 \
    --output_dir eval_results/relpose/scannet_s3_90/ttt3r_random \
    2>&1 | tee -a "$LOG"
log "DONE: ScanNet90 ttt3r_random"

# ttt3r_momentum_inv_t1 (brake)
log "START: ScanNet90 brake"
accelerate launch --num_processes 1 --main_process_port $PORT \
    eval/relpose/launch.py $COMMON_90 \
    --model_update_type ttt3r_momentum --momentum_tau 1.0 \
    --output_dir eval_results/relpose/scannet_s3_90/ttt3r_momentum_inv_t1 \
    2>&1 | tee -a "$LOG"
log "DONE: ScanNet90 brake"

# ttt3r_ortho
log "START: ScanNet90 ortho"
accelerate launch --num_processes 1 --main_process_port $PORT \
    eval/relpose/launch.py $COMMON_90 \
    --model_update_type ttt3r_ortho \
    --ortho_alpha_novel 0.5 --ortho_alpha_drift 0.05 --ortho_beta 0.95 \
    --output_dir eval_results/relpose/scannet_s3_90/ttt3r_ortho \
    2>&1 | tee -a "$LOG"
log "DONE: ScanNet90 ortho"

# ttt3r_ortho_adaptive (linear)
log "START: ScanNet90 ortho adaptive linear"
accelerate launch --num_processes 1 --main_process_port $PORT \
    eval/relpose/launch.py $COMMON_90 \
    --model_update_type ttt3r_ortho \
    --ortho_alpha_novel 0.5 --ortho_alpha_drift 0.05 --ortho_beta 0.95 \
    --ortho_adaptive linear \
    --output_dir eval_results/relpose/scannet_s3_90/ttt3r_ortho_adaptive \
    2>&1 | tee -a "$LOG"
log "DONE: ScanNet90 ortho adaptive linear"

# ============================================================
# 2. Sintel Relpose Eval — missing configs (~4 configs x 15min = 1h)
# ============================================================
COMMON_SIN="--weights $WEIGHTS --eval_dataset sintel --size 512"

# ttt3r_random p=0.5
log "START: Sintel ttt3r_random"
accelerate launch --num_processes 1 --main_process_port $PORT \
    eval/relpose/launch.py $COMMON_SIN \
    --model_update_type ttt3r_random --random_gate_p 0.5 \
    --output_dir eval_results/relpose/sintel/ttt3r_random \
    2>&1 | tee -a "$LOG"
log "DONE: Sintel ttt3r_random"

# ttt3r_momentum_inv_t1 (brake)
log "START: Sintel brake"
accelerate launch --num_processes 1 --main_process_port $PORT \
    eval/relpose/launch.py $COMMON_SIN \
    --model_update_type ttt3r_momentum --momentum_tau 1.0 \
    --output_dir eval_results/relpose/sintel/ttt3r_momentum_inv_t1 \
    2>&1 | tee -a "$LOG"
log "DONE: Sintel brake"

# ttt3r_ortho
log "START: Sintel ortho"
accelerate launch --num_processes 1 --main_process_port $PORT \
    eval/relpose/launch.py $COMMON_SIN \
    --model_update_type ttt3r_ortho \
    --ortho_alpha_novel 0.5 --ortho_alpha_drift 0.05 --ortho_beta 0.95 \
    --output_dir eval_results/relpose/sintel/ttt3r_ortho \
    2>&1 | tee -a "$LOG"
log "DONE: Sintel ortho"

# ttt3r_ortho_adaptive (linear)
log "START: Sintel ortho adaptive"
accelerate launch --num_processes 1 --main_process_port $PORT \
    eval/relpose/launch.py $COMMON_SIN \
    --model_update_type ttt3r_ortho \
    --ortho_alpha_novel 0.5 --ortho_alpha_drift 0.05 --ortho_beta 0.95 \
    --ortho_adaptive linear \
    --output_dir eval_results/relpose/sintel/ttt3r_ortho_adaptive \
    2>&1 | tee -a "$LOG"
log "DONE: Sintel ortho adaptive"

log "=== GPU1 overnight DONE ==="
