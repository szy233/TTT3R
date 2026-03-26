#!/bin/bash
# S3: Hyperparameter Sensitivity on ScanNet relpose
# Sweep spectral_temperature, geo_gate_tau, geo_gate_freq_cutoff
# Default: spectral_temperature=1.0, geo_gate_tau=2.0, geo_gate_freq_cutoff=4

set -e
export PYTHONPATH=src
GPU=${1:-0}
WEIGHTS="model/cut3r_512_dpt_4_64.pth"
PORT=29570

echo "=========================================="
echo "S3 Hyperparameter Sensitivity on GPU $GPU"
echo "=========================================="

# ── Sweep spectral_temperature (fix geo_tau=2, cutoff=4) ──
for st in 0.5 1.0 2.0 4.0; do
    TAG="ttt3r_joint_st${st}_gt2.0_c4"
    OUT="eval_results/relpose/scannet_s3_1000/$TAG"
    if [ -f "$OUT/_error_log.txt" ] && [ "$(grep -c '| ATE' "$OUT/_error_log.txt" 2>/dev/null)" -ge 60 ]; then
        echo "[SKIP] $TAG already done"
        continue
    fi
    echo ""
    echo "[$(date)] $TAG"
    CUDA_VISIBLE_DEVICES=$GPU accelerate launch --num_processes 1 --main_process_port $PORT \
        eval/relpose/launch.py \
        --weights $WEIGHTS --output_dir $OUT \
        --eval_dataset scannet_s3_1000 --size 512 --model_update_type ttt3r_joint \
        --spectral_temperature $st --geo_gate_tau 2.0 --geo_gate_freq_cutoff 4
    PORT=$((PORT + 1))
    echo "[$(date)] Done $TAG"
done

# ── Sweep geo_gate_tau (fix st=1.0, cutoff=4) ──
for gt in 0.5 1.0 2.0 4.0; do
    TAG="ttt3r_joint_st1.0_gt${gt}_c4"
    OUT="eval_results/relpose/scannet_s3_1000/$TAG"
    if [ -f "$OUT/_error_log.txt" ] && [ "$(grep -c '| ATE' "$OUT/_error_log.txt" 2>/dev/null)" -ge 60 ]; then
        echo "[SKIP] $TAG already done"
        continue
    fi
    echo ""
    echo "[$(date)] $TAG"
    CUDA_VISIBLE_DEVICES=$GPU accelerate launch --num_processes 1 --main_process_port $PORT \
        eval/relpose/launch.py \
        --weights $WEIGHTS --output_dir $OUT \
        --eval_dataset scannet_s3_1000 --size 512 --model_update_type ttt3r_joint \
        --spectral_temperature 1.0 --geo_gate_tau $gt --geo_gate_freq_cutoff 4
    PORT=$((PORT + 1))
    echo "[$(date)] Done $TAG"
done

# ── Sweep freq_cutoff (fix st=1.0, gt=2.0) ──
for c in 2 4 8; do
    TAG="ttt3r_joint_st1.0_gt2.0_c${c}"
    OUT="eval_results/relpose/scannet_s3_1000/$TAG"
    if [ -f "$OUT/_error_log.txt" ] && [ "$(grep -c '| ATE' "$OUT/_error_log.txt" 2>/dev/null)" -ge 60 ]; then
        echo "[SKIP] $TAG already done"
        continue
    fi
    echo ""
    echo "[$(date)] $TAG"
    CUDA_VISIBLE_DEVICES=$GPU accelerate launch --num_processes 1 --main_process_port $PORT \
        eval/relpose/launch.py \
        --weights $WEIGHTS --output_dir $OUT \
        --eval_dataset scannet_s3_1000 --size 512 --model_update_type ttt3r_joint \
        --spectral_temperature 1.0 --geo_gate_tau 2.0 --geo_gate_freq_cutoff $c
    PORT=$((PORT + 1))
    echo "[$(date)] Done $TAG"
done

echo ""
echo "=========================================="
echo "[$(date)] S3 complete!"
echo "=========================================="
