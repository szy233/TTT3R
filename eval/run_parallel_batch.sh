#!/bin/bash
# =============================================================================
# Parallel batch launcher: maximize GPU utilization
# Each experiment ~6G VRAM. Unique port per process.
# =============================================================================

cd /home/szy/research/TTT3R

LOGDIR="eval_logs/batch_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

PORT_BASE=30000
PORT_IDX=0

launch() {
    local GPU=$1 DATASET=$2 METHOD=$3
    local PORT=$((PORT_BASE + PORT_IDX))
    PORT_IDX=$((PORT_IDX + 1))

    local LOGFILE="${LOGDIR}/${DATASET}_${METHOD}_gpu${GPU}.log"
    echo "[LAUNCH] GPU=$GPU  $DATASET / $METHOD  port=$PORT  log=$LOGFILE"

    DDD3R_PORT_OFFSET=$((PORT - 29560 - GPU)) \
    bash eval/run_ddd3r_eval.sh "$GPU" "$DATASET" "$METHOD" \
        > "$LOGFILE" 2>&1 &
}

METHODS="cut3r ttt3r ddd3r_constant ddd3r_brake ddd3r ddd3r_entropy"

# --- GPU 0 (~73G free, 11 slots) ---
# ScanNet scaling curve: 200f × 6 methods
for m in $METHODS; do
    launch 0 scannet_s3_200 "$m"
done

# ScanNet scaling curve: 500f × 5 methods
for m in cut3r ttt3r ddd3r_constant ddd3r_brake ddd3r; do
    launch 0 scannet_s3_500 "$m"
done

# --- GPU 1 (~53G free, 8 slots) ---
# ScanNet 500f remaining
launch 1 scannet_s3_500 ddd3r_entropy

# Video depth: entropy on kitti & bonn
launch 1 kitti ddd3r_entropy
launch 1 bonn ddd3r_entropy

# Video depth: new-naming methods on sintel_depth (only has entropy so far)
launch 1 sintel_depth ddd3r_constant
launch 1 sintel_depth ddd3r_brake
launch 1 sintel_depth ddd3r

# Video depth: new-naming on kitti & bonn
launch 1 kitti ddd3r_constant
launch 1 bonn ddd3r_constant

echo ""
echo "=== Launched $PORT_IDX experiments ==="
echo "Logs: $LOGDIR/"
echo "Monitor: tail -f $LOGDIR/*.log"
echo "Check GPU: nvidia-smi"
echo "Wait all: wait"

wait
echo "=== ALL DONE ==="
