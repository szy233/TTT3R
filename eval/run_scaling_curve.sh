#!/bin/bash
# =============================================================================
# Scaling Curve — TTT3R standard frame configs, 2-GPU parallel
#
# ScanNet: 50,90,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000
# TUM:     50,100,150,200,300,400,500,600,700,800,900,1000
# Methods: cut3r, ttt3r, ddd3r_constant, ddd3r_brake
#
# Usage: bash eval/run_scaling_curve.sh <GPU0> <GPU1>
# =============================================================================

set -e
cd /home/szy/research/TTT3R

GPU0=${1:-0}
GPU1=${2:-1}

METHODS=(cut3r ttt3r ddd3r_constant ddd3r_brake)

SCANNET_FRAMES=(50 90 100 150 200 250 300 350 400 450 500 550 600 650 700 750 800 850 900 950 1000)
TUM_FRAMES=(50 100 150 200 300 400 500 600 700 800 900 1000)

# Build full job list
ALL_JOBS=()
for METHOD in "${METHODS[@]}"; do
    for N in "${SCANNET_FRAMES[@]}"; do
        ALL_JOBS+=("scannet_s3_${N} ${METHOD}")
    done
    for N in "${TUM_FRAMES[@]}"; do
        ALL_JOBS+=("tum_s1_${N} ${METHOD}")
    done
done

# Filter out completed jobs
JOBS=()
for JOB in "${ALL_JOBS[@]}"; do
    DATASET=$(echo "$JOB" | awk '{print $1}')
    METHOD=$(echo "$JOB" | awk '{print $2}')
    OUTDIR="eval_results/relpose/${DATASET}/${METHOD}"
    if ls "$OUTDIR"/*_eval_metric.txt &>/dev/null 2>&1; then
        echo "[SKIP] ${DATASET} ${METHOD}"
    else
        JOBS+=("$JOB")
    fi
done

TOTAL=${#JOBS[@]}
echo "=== Scaling Curve: ${TOTAL} jobs (${#ALL_JOBS[@]} total, $((${#ALL_JOBS[@]} - TOTAL)) skipped) ==="

# Split jobs: even indices → GPU0, odd indices → GPU1
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
echo "Monitor: tail -f /proc/${PID0}/fd/1 or watch nvidia-smi"
wait $PID0 $PID1

echo "=== All scaling curve jobs complete ==="
