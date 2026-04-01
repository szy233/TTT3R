#!/bin/bash
# =============================================================================
# Auto-gamma experiment: 6 methods × 4 datasets
# Uses run_ddd3r_eval.sh for each (method, dataset) pair
#
# Usage: bash eval/run_auto_gamma_eval.sh [GPU_LIST]
#   GPU_LIST: comma-separated GPU IDs (default: 0,1)
#   e.g.: bash eval/run_auto_gamma_eval.sh 0,1
#         bash eval/run_auto_gamma_eval.sh 0
#         bash eval/run_auto_gamma_eval.sh 0,1,2,3
# =============================================================================
set -e
cd "$(dirname "$0")/.."

IFS=',' read -ra GPUS <<< "${1:-0,1}"
NUM_GPUS=${#GPUS[@]}

METHODS=(
    ddd3r_auto_warmup_linear
    ddd3r_auto_warmup_threshold
    ddd3r_auto_steep_sigmoid
    ddd3r_auto_steep_sigmoid_k20
    ddd3r_auto_steep_clamp
    ddd3r_auto_steep_clamp_tight
)

DATASETS=(
    tum_s1_1000
    scannet_s3_1000
    kitti_odom
)

# Build job list: (method, dataset) pairs
JOBS=()
for method in "${METHODS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        JOBS+=("${method}|${dataset}")
    done
done

TOTAL=${#JOBS[@]}
echo "=== Auto-gamma Eval: ${TOTAL} jobs on ${NUM_GPUS} GPU(s) ==="
echo "  GPUs: ${GPUS[*]}"
echo "  Methods: ${METHODS[*]}"
echo "  Datasets: ${DATASETS[*]}"
echo ""

# Track PIDs per GPU slot
declare -A GPU_PID
LOG_DIR="eval_results/_auto_gamma_logs"
mkdir -p "$LOG_DIR"

wait_for_gpu() {
    local slot=$1
    local pid=${GPU_PID[$slot]:-}
    if [ -n "$pid" ]; then
        wait "$pid" 2>/dev/null || true
    fi
}

JOB_IDX=0
for job in "${JOBS[@]}"; do
    IFS='|' read -r method dataset <<< "$job"
    SLOT=$((JOB_IDX % NUM_GPUS))
    GPU=${GPUS[$SLOT]}

    # Wait for this GPU slot to be free
    wait_for_gpu $SLOT

    LOG_FILE="${LOG_DIR}/${method}_${dataset}.log"
    echo "[${JOB_IDX}/${TOTAL}] GPU${GPU}: ${method} on ${dataset} → ${LOG_FILE}"

    bash eval/run_ddd3r_eval.sh "$GPU" "$dataset" "$method" > "$LOG_FILE" 2>&1 &
    GPU_PID[$SLOT]=$!

    JOB_IDX=$((JOB_IDX + 1))
done

# Wait for all remaining jobs
for slot in $(seq 0 $((NUM_GPUS - 1))); do
    wait_for_gpu $slot
done

echo ""
echo "=== All ${TOTAL} jobs complete ==="
echo ""

# Print summary: grep ATE/abs_rel from logs
echo "=== Results Summary ==="
for job in "${JOBS[@]}"; do
    IFS='|' read -r method dataset <<< "$job"
    LOG_FILE="${LOG_DIR}/${method}_${dataset}.log"
    if [ -f "$LOG_FILE" ]; then
        # Try to extract key metric from log
        RESULT=$(grep -E "(ATE|Mean ATE|abs_rel|mean_ate)" "$LOG_FILE" | tail -1 || echo "  (check log)")
        echo "${method} | ${dataset}: ${RESULT}"
    else
        echo "${method} | ${dataset}: MISSING LOG"
    fi
done
