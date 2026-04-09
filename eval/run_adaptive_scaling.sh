#!/bin/bash
# Adaptive methods scaling curve for analysis section
# Usage: bash eval/run_adaptive_scaling.sh <GPU0> <GPU1>
# Runs local_de_raw_p2 and fmean_sig on key scaling curve points
# Chain after main scaling curve: wait for PID, then run

set -e

GPU0=${1:-0}
GPU1=${2:-1}

METHODS="ddd3r_local_de_raw_p2 ddd3r_fmean_sig"

# ScanNet: key points (8 points for smooth curve)
SCANNET_FRAMES="50 90 200 350 500 700 850 1000"
# TUM: key points (6 points)
TUM_FRAMES="50 200 500 700 900 1000"

# Collect all jobs
JOBS=()
for M in $METHODS; do
    for N in $SCANNET_FRAMES; do
        JOBS+=("scannet_s3_${N} ${M}")
    done
    for N in $TUM_FRAMES; do
        JOBS+=("tum_s1_${N} ${M}")
    done
done

TOTAL=${#JOBS[@]}
echo "=== Adaptive scaling curve: $TOTAL jobs, 2 GPUs ==="
echo "Methods: $METHODS"
echo "ScanNet frames: $SCANNET_FRAMES"
echo "TUM frames: $TUM_FRAMES"
echo ""

# Split into two queues (even/odd)
run_queue() {
    local GPU=$1
    local START=$2
    local STEP=$3
    local PORT_OFFSET=$((GPU * 100 + 400))

    for ((i=START; i<TOTAL; i+=STEP)); do
        DS=$(echo "${JOBS[$i]}" | cut -d' ' -f1)
        METHOD=$(echo "${JOBS[$i]}" | cut -d' ' -f2)

        # Skip if result already exists with enough scenes
        RESULT_DIR="eval_results/relpose/${DS}/${METHOD}"
        if [ -d "$RESULT_DIR" ]; then
            EXPECTED=8  # TUM
            [[ "$DS" == scannet* ]] && EXPECTED=60
            N_DONE=$(find "$RESULT_DIR" -maxdepth 1 -name "*_eval_metric.txt" 2>/dev/null | wc -l)
            if [ "$N_DONE" -ge "$EXPECTED" ]; then
                echo "[GPU${GPU}] SKIP $DS $METHOD ($N_DONE scenes exist)"
                continue
            fi
        fi

        echo "[GPU${GPU}] $DS $METHOD"
        DDD3R_PORT_OFFSET=$PORT_OFFSET bash eval/run_ddd3r_eval.sh $GPU $DS $METHOD 2>&1 | tail -2
    done
}

# Run two queues in parallel
run_queue $GPU0 0 2 &
PID0=$!
run_queue $GPU1 1 2 &
PID1=$!

echo "GPU${GPU0} PID: $PID0, GPU${GPU1} PID: $PID1"
wait $PID0 $PID1
echo "=== Adaptive scaling curve DONE ==="
