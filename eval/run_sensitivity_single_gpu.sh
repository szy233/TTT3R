#!/bin/bash
set -e

cd "$(dirname "$0")/.."
export PYTHONPATH=$(pwd)/src

MODEL_WEIGHTS="./model/cut3r_512_dpt_4_64.pth"
UPDATE_TYPE="ttt3r_joint"
SPECTRAL_TAU=1.0

# low-VRAM setting
DATASET="tum_s1_50"
IMG_SIZE=384
GPU_ID=0

TAUS=(0.5 1.0 2.0 4.0)
CUTOFFS=(2 4 8)

BASE_OUT="eval_results/relpose/sensitivity/tum"

mkdir -p "$BASE_OUT"

echo "=========================================="
echo "Single-GPU sensitivity sweep"
echo "GPU: ${GPU_ID}"
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

        if [ -f "${output_dir}/rgbd_dataset_freiburg3_long_office_household_eval_metric.txt" ]; then
            echo "[SKIP] ${tag}"
            continue
        fi

        echo "[RUN ] ${tag}"

        CUDA_VISIBLE_DEVICES=${GPU_ID} python eval/relpose/launch.py \
            --weights "${MODEL_WEIGHTS}" \
            --output_dir "${output_dir}" \
            --eval_dataset "${DATASET}" \
            --size "${IMG_SIZE}" \
            --model_update_type "${UPDATE_TYPE}" \
            --spectral_temperature "${SPECTRAL_TAU}" \
            --geo_gate_tau "${tau}" \
            --geo_gate_freq_cutoff "${cutoff}"

        echo "[DONE] ${tag}"
        echo ""
    done
done

echo "=========================================="
echo "Sweep complete."
echo "Now run:"
echo "python analysis/sensitivity_analysis.py"
echo "=========================================="