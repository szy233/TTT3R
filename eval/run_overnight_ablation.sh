#!/bin/bash
# Overnight ablation experiments
# Usage: bash eval/run_overnight_ablation.sh <GPU0> <GPU1>
# Runs: α∥ ablation + momentum decomposition + novel boost + proj_frac ScanNet

set -e
GPU0=${1:-0}
GPU1=${2:-1}
PORT0=950
PORT1=960

echo "=== Overnight ablation experiments ==="
echo "GPU0=$GPU0, GPU1=$GPU1"
date

# Priority 1: α∥ ablation on both datasets (most important)
# α∥ = 0.10, 0.15, 0.20, 0.25 (current default is 0.05)
echo ""
echo "=== Priority 1: α∥ ablation ==="

run_gpu0() {
    # GPU0: ScanNet jobs (slower)
    for AP in 10 15 20 25; do
        echo "[GPU$GPU0] ddd3r_a${AP} scannet_s3_1000"
        DDD3R_PORT_OFFSET=$PORT0 bash eval/run_ddd3r_eval.sh $GPU0 scannet_s3_1000 ddd3r_a${AP} 2>&1 | tail -2
    done
    # GPU0: momentum + proj_frac ScanNet
    echo "[GPU$GPU0] ddd3r_momentum scannet_s3_1000"
    DDD3R_PORT_OFFSET=$PORT0 bash eval/run_ddd3r_eval.sh $GPU0 scannet_s3_1000 ddd3r_momentum 2>&1 | tail -2
    echo "[GPU$GPU0] ddd3r_proj_frac scannet_s3_1000"
    DDD3R_PORT_OFFSET=$PORT0 bash eval/run_ddd3r_eval.sh $GPU0 scannet_s3_1000 ddd3r_proj_frac 2>&1 | tail -2
    # GPU0: novel boost ScanNet
    for G in 02 05 10 15; do
        echo "[GPU$GPU0] ddd3r_boost${G} scannet_s3_1000"
        DDD3R_PORT_OFFSET=$PORT0 bash eval/run_ddd3r_eval.sh $GPU0 scannet_s3_1000 ddd3r_boost${G} 2>&1 | tail -2
    done
}

run_gpu1() {
    # GPU1: TUM jobs (faster) + then ScanNet overflow
    for AP in 10 15 20 25; do
        echo "[GPU$GPU1] ddd3r_a${AP} tum_s1_1000"
        DDD3R_PORT_OFFSET=$PORT1 bash eval/run_ddd3r_eval.sh $GPU1 tum_s1_1000 ddd3r_a${AP} 2>&1 | tail -2
    done
    # GPU1: momentum + novel boost TUM
    echo "[GPU$GPU1] ddd3r_momentum tum_s1_1000"
    DDD3R_PORT_OFFSET=$PORT1 bash eval/run_ddd3r_eval.sh $GPU1 tum_s1_1000 ddd3r_momentum 2>&1 | tail -2
    for G in 02 05 10 15; do
        echo "[GPU$GPU1] ddd3r_boost${G} tum_s1_1000"
        DDD3R_PORT_OFFSET=$PORT1 bash eval/run_ddd3r_eval.sh $GPU1 tum_s1_1000 ddd3r_boost${G} 2>&1 | tail -2
    done
}

run_gpu0 &
PID0=$!
run_gpu1 &
PID1=$!

echo "GPU$GPU0 PID: $PID0, GPU$GPU1 PID: $PID1"
wait $PID0 $PID1

echo ""
echo "=== ALL OVERNIGHT EXPERIMENTS DONE ==="
date
