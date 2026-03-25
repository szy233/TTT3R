#!/bin/bash
set -u

cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)/src"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

MODEL_WEIGHTS="./model/cut3r_512_dpt_4_64.pth"
UPDATE_TYPE="ttt3r_joint"
SPECTRAL_TAU=1.0

DATASET="tum_s1_1000"
IMG_SIZE=512

TAUS=(0.5 1.0 2.0 4.0)
CUTOFFS=(2 4 8)

BASE_OUT="eval_results/relpose/sensitivity/tum"

mkdir -p "$BASE_OUT"

echo "=========================================="
echo "TUM sensitivity sweep (formal setting)"
echo "GPU: 0"
echo "Dataset: ${DATASET}"
echo "Image size: ${IMG_SIZE}"
echo "Model: ${UPDATE_TYPE}"
echo "Tau grid: ${TAUS[*]}"
echo "Cutoff grid: ${CUTOFFS[*]}"
echo "Results: ${BASE_OUT}"
echo "=========================================="

for tau in "${TAUS[@]}"; do
    for cutoff in "${CUTOFFS[@]}"; do
        tag="${UPDATE_TYPE}_tau${tau}_c${cutoff}"
        output_dir="${BASE_OUT}/${tag}"
        metric_file="${output_dir}/rgbd_dataset_freiburg3_long_office_household_eval_metric.txt"
        fail_log="${output_dir}/FAILED.txt"

        mkdir -p "${output_dir}"

        if [ -f "${metric_file}" ]; then
            echo "[SKIP] ${tag}"
            continue
        fi

        rm -f "${fail_log}"

        echo "[RUN ] ${tag}"
        python eval/relpose/launch.py \
            --weights "${MODEL_WEIGHTS}" \
            --output_dir "${output_dir}" \
            --eval_dataset "${DATASET}" \
            --size "${IMG_SIZE}" \
            --model_update_type "${UPDATE_TYPE}" \
            --spectral_temperature "${SPECTRAL_TAU}" \
            --geo_gate_tau "${tau}" \
            --geo_gate_freq_cutoff "${cutoff}"

        status=$?
        if [ $status -ne 0 ]; then
            echo "[FAIL] ${tag} (exit=${status})"
            echo "exit=${status}" > "${fail_log}"
        else
            echo "[DONE] ${tag}"
        fi

        echo ""
        sleep 3
    done
done

echo "=========================================="
echo "Sweep complete."
echo "Now run:"
echo "python analysis/sensitivity_analysis_tum1000.py"
echo "=========================================="