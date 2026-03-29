#!/bin/bash
# GPU0 overnight experiments: Inference overhead + ScanNet sensitivity
# 2026-03-28 晚
set -e
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=src
export PYTHONUNBUFFERED=1

WEIGHTS="model/cut3r_512_dpt_4_64.pth"
PORT=29570
LOG="eval_results/overnight_gpu0.log"
mkdir -p eval_results

log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG"; }

log "=== GPU0 overnight started ==="

# ============================================================
# 1. Inference Overhead Benchmark (~20min)
# ============================================================
log "START: Inference overhead benchmark"
python eval/benchmark_overhead.py \
    --weights $WEIGHTS --size 512 \
    --eval_dataset tum_s1_1000 \
    --max_frames 200 --max_seqs 3 \
    --warmup 1 --repeats 3 \
    --output eval_results/benchmark_overhead.json \
    2>&1 | tee -a "$LOG"
log "DONE: Inference overhead benchmark"

# ============================================================
# 2. ScanNet Ortho Hyperparameter Sensitivity (~7x4h = 28h... too long)
#    Use scannet_s3_90 (short) instead for faster turnaround (~7x20min = 2.5h)
# ============================================================
COMMON="--weights $WEIGHTS --eval_dataset scannet_s3_90 --size 512"

# Default ortho (α_novel=0.5, α_drift=0.05, β=0.95)
log "START: ScanNet90 ortho default"
accelerate launch --num_processes 1 --main_process_port $PORT \
    eval/relpose/launch.py $COMMON \
    --model_update_type ttt3r_ortho \
    --ortho_alpha_novel 0.5 --ortho_alpha_drift 0.05 --ortho_beta 0.95 \
    --output_dir eval_results/relpose/scannet_s3_90/ttt3r_ortho_an05_ad005 \
    2>&1 | tee -a "$LOG"
log "DONE: ScanNet90 ortho default"

# α_drift=0.1
log "START: ScanNet90 ortho α_drift=0.1"
accelerate launch --num_processes 1 --main_process_port $PORT \
    eval/relpose/launch.py $COMMON \
    --model_update_type ttt3r_ortho \
    --ortho_alpha_novel 0.5 --ortho_alpha_drift 0.1 --ortho_beta 0.95 \
    --output_dir eval_results/relpose/scannet_s3_90/ttt3r_ortho_an05_ad0.1 \
    2>&1 | tee -a "$LOG"
log "DONE: ScanNet90 ortho α_drift=0.1"

# α_drift=0.2
log "START: ScanNet90 ortho α_drift=0.2"
accelerate launch --num_processes 1 --main_process_port $PORT \
    eval/relpose/launch.py $COMMON \
    --model_update_type ttt3r_ortho \
    --ortho_alpha_novel 0.5 --ortho_alpha_drift 0.2 --ortho_beta 0.95 \
    --output_dir eval_results/relpose/scannet_s3_90/ttt3r_ortho_an05_ad0.2 \
    2>&1 | tee -a "$LOG"
log "DONE: ScanNet90 ortho α_drift=0.2"

# α_novel=0.7
log "START: ScanNet90 ortho α_novel=0.7"
accelerate launch --num_processes 1 --main_process_port $PORT \
    eval/relpose/launch.py $COMMON \
    --model_update_type ttt3r_ortho \
    --ortho_alpha_novel 0.7 --ortho_alpha_drift 0.05 --ortho_beta 0.95 \
    --output_dir eval_results/relpose/scannet_s3_90/ttt3r_ortho_an0.7_ad005 \
    2>&1 | tee -a "$LOG"
log "DONE: ScanNet90 ortho α_novel=0.7"

# α_novel=0.3
log "START: ScanNet90 ortho α_novel=0.3"
accelerate launch --num_processes 1 --main_process_port $PORT \
    eval/relpose/launch.py $COMMON \
    --model_update_type ttt3r_ortho \
    --ortho_alpha_novel 0.3 --ortho_alpha_drift 0.05 --ortho_beta 0.95 \
    --output_dir eval_results/relpose/scannet_s3_90/ttt3r_ortho_an0.3_ad005 \
    2>&1 | tee -a "$LOG"
log "DONE: ScanNet90 ortho α_novel=0.3"

# β=0.9
log "START: ScanNet90 ortho β=0.9"
accelerate launch --num_processes 1 --main_process_port $PORT \
    eval/relpose/launch.py $COMMON \
    --model_update_type ttt3r_ortho \
    --ortho_alpha_novel 0.5 --ortho_alpha_drift 0.05 --ortho_beta 0.9 \
    --output_dir eval_results/relpose/scannet_s3_90/ttt3r_ortho_an05_ad005_b0.9 \
    2>&1 | tee -a "$LOG"
log "DONE: ScanNet90 ortho β=0.9"

# β=0.99
log "START: ScanNet90 ortho β=0.99"
accelerate launch --num_processes 1 --main_process_port $PORT \
    eval/relpose/launch.py $COMMON \
    --model_update_type ttt3r_ortho \
    --ortho_alpha_novel 0.5 --ortho_alpha_drift 0.05 --ortho_beta 0.99 \
    --output_dir eval_results/relpose/scannet_s3_90/ttt3r_ortho_an05_ad005_b0.99 \
    2>&1 | tee -a "$LOG"
log "DONE: ScanNet90 ortho β=0.99"

log "=== GPU0 overnight DONE ==="
