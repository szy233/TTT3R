#!/bin/bash
# =============================================================================
# DDD3R (ortho) Scaling Curve — supplement to main scaling curve
# Runs ddd3r method across all frame configs for relpose + video depth
#
# Usage: bash eval/run_ddd3r_ortho_scaling.sh <GPU0> <GPU1>
# =============================================================================

set -e
cd /home/szy/research/TTT3R

GPU0=${1:-0}
GPU1=${2:-1}

METHOD=ddd3r

SCANNET_FRAMES=(50 90 100 150 200 250 300 350 400 450 500 550 600 650 700 750 800 850 900 950 1000)
TUM_FRAMES=(50 100 150 200 300 400 500 600 700 800 900 1000)
KITTI_FRAMES=(50 100 150 200 250 300 350 400 450 500)
BONN_FRAMES=(50 100 150 200 250 300 350 400 450 500)

# Build job list, skip existing
JOBS=()

# Relpose
for N in "${SCANNET_FRAMES[@]}"; do
    OUTDIR="eval_results/relpose/scannet_s3_${N}/${METHOD}"
    if ls "$OUTDIR"/*_eval_metric.txt &>/dev/null 2>&1; then
        echo "[SKIP] scannet_s3_${N} ${METHOD}"
    else
        JOBS+=("scannet_s3_${N} ${METHOD}")
    fi
done
for N in "${TUM_FRAMES[@]}"; do
    OUTDIR="eval_results/relpose/tum_s1_${N}/${METHOD}"
    if ls "$OUTDIR"/*_eval_metric.txt &>/dev/null 2>&1; then
        echo "[SKIP] tum_s1_${N} ${METHOD}"
    else
        JOBS+=("tum_s1_${N} ${METHOD}")
    fi
done

# Video depth
for N in "${KITTI_FRAMES[@]}"; do
    OUTDIR="eval_results/video_depth/kitti_s1_${N}/${METHOD}"
    if [ -d "$OUTDIR" ]; then
        echo "[SKIP] kitti_s1_${N} ${METHOD}"
    else
        JOBS+=("kitti_s1_${N} ${METHOD}")
    fi
done
for N in "${BONN_FRAMES[@]}"; do
    OUTDIR="eval_results/video_depth/bonn_s1_${N}/${METHOD}"
    if [ -d "$OUTDIR" ]; then
        echo "[SKIP] bonn_s1_${N} ${METHOD}"
    else
        JOBS+=("bonn_s1_${N} ${METHOD}")
    fi
done
# Sintel
OUTDIR="eval_results/video_depth/sintel_depth/${METHOD}"
if [ -d "$OUTDIR" ]; then
    echo "[SKIP] sintel_depth ${METHOD}"
else
    JOBS+=("sintel_depth ${METHOD}")
fi

TOTAL=${#JOBS[@]}
echo "=== DDD3R Ortho Scaling: ${TOTAL} jobs ==="

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

run_queue "$GPU0" "${GPU0_JOBS[@]}" &
PID0=$!
run_queue "$GPU1" "${GPU1_JOBS[@]}" &
PID1=$!

echo "Launched: GPU${GPU0}(pid=${PID0}) GPU${GPU1}(pid=${PID1})"
wait $PID0 $PID1

echo "=== All ddd3r ortho scaling jobs complete ==="
