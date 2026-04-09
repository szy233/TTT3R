#!/bin/bash
# =============================================================================
# Video Depth Scaling Curve — TTT3R standard frame configs, 2-GPU parallel
#
# KITTI: 50,100,150,200,250,300,350,400,450,500
# Bonn:  50,100,150,200,250,300,350,400,450,500
# Sintel: fixed (no scaling)
# Methods: cut3r, ttt3r, ddd3r_constant, ddd3r_brake
# Align: scale&shift (default, matches CUT3R/MonST3R protocol)
#
# Usage: bash eval/run_vdepth_scaling_curve.sh <GPU0> <GPU1>
# =============================================================================

set -e
cd /home/szy/research/TTT3R

GPU0=${1:-0}
GPU1=${2:-1}

METHODS=(cut3r ttt3r ddd3r_constant ddd3r_brake)
KITTI_FRAMES=(50 100 150 200 250 300 350 400 450 500)
BONN_FRAMES=(50 100 150 200 250 300 350 400 450 500)

# Build full job list
ALL_JOBS=()
for METHOD in "${METHODS[@]}"; do
    for N in "${KITTI_FRAMES[@]}"; do
        ALL_JOBS+=("kitti_s1_${N} ${METHOD}")
    done
    for N in "${BONN_FRAMES[@]}"; do
        ALL_JOBS+=("bonn_s1_${N} ${METHOD}")
    done
    # Sintel (fixed length)
    ALL_JOBS+=("sintel_depth ${METHOD}")
done

# Filter completed jobs (check for depth metric output)
JOBS=()
for JOB in "${ALL_JOBS[@]}"; do
    DATASET=$(echo "$JOB" | awk '{print $1}')
    METHOD=$(echo "$JOB" | awk '{print $2}')
    OUTDIR="eval_results/video_depth/${DATASET}/${METHOD}"
    # Check if depth eval results exist (abs_rel file from eval_depth.py)
    if [ -f "$OUTDIR/depth_metrics_scale_shift.txt" ] 2>/dev/null; then
        echo "[SKIP] ${DATASET} ${METHOD}"
    elif [ -f "$OUTDIR/result_metric.json" ] 2>/dev/null; then
        echo "[SKIP] ${DATASET} ${METHOD}"
    elif ls "$OUTDIR"/*_eval_metric.txt &>/dev/null 2>&1; then
        echo "[SKIP] ${DATASET} ${METHOD}"
    else
        JOBS+=("$JOB")
    fi
done

TOTAL=${#JOBS[@]}
echo "=== Video Depth Scaling: ${TOTAL} jobs (${#ALL_JOBS[@]} total, $((${#ALL_JOBS[@]} - TOTAL)) skipped) ==="

# Split jobs between GPUs
GPU0_JOBS=()
GPU1_JOBS=()
for i in "${!JOBS[@]}"; do
    if (( i % 2 == 0 )); then
        GPU0_JOBS+=("${JOBS[$i]}")
    else
        GPU1_JOBS+=("${JOBS[$i]}")
    fi
done

echo "GPU${GPU0}: ${#GPU0_JOBS[@]} jobs | GPU${GPU1}: ${#GPU1_JOBS[@]} jobs"

run_queue() {
    local GPU=$1
    shift
    local QUEUE=("$@")
    local IDX=0
    for JOB in "${QUEUE[@]}"; do
        IDX=$((IDX+1))
        local DATASET=$(echo "$JOB" | awk '{print $1}')
        local METHOD=$(echo "$JOB" | awk '{print $2}')
        echo "[GPU${GPU}] (${IDX}/${#QUEUE[@]}) ${DATASET} ${METHOD}"
        bash eval/run_ddd3r_eval.sh "$GPU" "$DATASET" "$METHOD" 2>&1 | tail -3
        echo "[GPU${GPU}] DONE: ${DATASET} ${METHOD}"
    done
    echo "[GPU${GPU}] All jobs complete."
}

# Launch two queues in parallel
run_queue "$GPU0" "${GPU0_JOBS[@]}" &
PID0=$!
run_queue "$GPU1" "${GPU1_JOBS[@]}" &
PID1=$!

echo "Launched: GPU${GPU0}(pid=${PID0}) GPU${GPU1}(pid=${PID1})"
wait $PID0 $PID1

echo "=== All video depth scaling jobs complete ==="
