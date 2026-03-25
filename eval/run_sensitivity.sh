#!/bin/bash
# =============================================================================
# TTT3R Hyperparameter Sensitivity Sweep
# Grid: geo_gate_tau ∈ {0.5, 1, 2, 4} × geo_gate_freq_cutoff ∈ {2, 4, 8}
# Model: ttt3r_joint (final method, L23+ttt3r)
# GPU0: ScanNet relpose (scannet_s3_1000)
# GPU1: TUM relpose    (tum_s1_1000)
#
# Usage:
#   nohup bash eval/run_sensitivity.sh > eval/sensitivity.log 2>&1 &
# =============================================================================
set -e
cd "$(dirname "$0")/.."

MODEL_WEIGHTS="model/cut3r_512_dpt_4_64.pth"
UPDATE_TYPE="ttt3r_joint"
SPECTRAL_TAU=1.0   # L2 temperature fixed (insensitive, use τ=1)

TAUS=(0.5 1.0 2.0 4.0)
CUTOFFS=(2 4 8)

mkdir -p eval_results/relpose/sensitivity

# ── GPU 0: ScanNet ──────────────────────────────────────────────────────────
run_scannet() {
    echo "[GPU0/ScanNet] Starting sensitivity sweep (${#TAUS[@]} × ${#CUTOFFS[@]} = $((${#TAUS[@]} * ${#CUTOFFS[@]})) configs)"
    for tau in "${TAUS[@]}"; do
        for cutoff in "${CUTOFFS[@]}"; do
            tag="${UPDATE_TYPE}_tau${tau}_c${cutoff}"
            output_dir="eval_results/relpose/sensitivity/scannet/${tag}"

            if [ -f "${output_dir}/_error_log.txt" ]; then
                echo "[GPU0/ScanNet] SKIP (exists): ${tag}"
                continue
            fi

            echo "[GPU0/ScanNet] Running: ${tag}"
            CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src accelerate launch \
                --num_processes 1 --main_process_port 29570 \
                eval/relpose/launch.py \
                --weights "$MODEL_WEIGHTS" \
                --output_dir "$output_dir" \
                --eval_dataset scannet_s3_1000 \
                --size 512 \
                --model_update_type "$UPDATE_TYPE" \
                --spectral_temperature $SPECTRAL_TAU \
                --geo_gate_tau "$tau" \
                --geo_gate_freq_cutoff "$cutoff"
            echo "[GPU0/ScanNet] ✓ ${tag}"
        done
    done
    echo "[GPU0] ScanNet sensitivity sweep done."
}

# ── GPU 1: TUM ──────────────────────────────────────────────────────────────
run_tum() {
    echo "[GPU1/TUM] Starting sensitivity sweep"
    for tau in "${TAUS[@]}"; do
        for cutoff in "${CUTOFFS[@]}"; do
            tag="${UPDATE_TYPE}_tau${tau}_c${cutoff}"
            output_dir="eval_results/relpose/sensitivity/tum/${tag}"

            if [ -f "${output_dir}/_error_log.txt" ]; then
                echo "[GPU1/TUM] SKIP (exists): ${tag}"
                continue
            fi

            echo "[GPU1/TUM] Running: ${tag}"
            CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src accelerate launch \
                --num_processes 1 --main_process_port 29571 \
                eval/relpose/launch.py \
                --weights "$MODEL_WEIGHTS" \
                --output_dir "$output_dir" \
                --eval_dataset tum_s1_1000 \
                --size 512 \
                --model_update_type "$UPDATE_TYPE" \
                --spectral_temperature $SPECTRAL_TAU \
                --geo_gate_tau "$tau" \
                --geo_gate_freq_cutoff "$cutoff"
            echo "[GPU1/TUM] ✓ ${tag}"
        done
    done
    echo "[GPU1] TUM sensitivity sweep done."
}

# ── Launch in parallel ───────────────────────────────────────────────────────
run_scannet &
PID_SCANNET=$!

run_tum &
PID_TUM=$!

echo "=========================================="
echo "Sensitivity sweep launched:"
echo "  GPU0 (ScanNet): PID ${PID_SCANNET}"
echo "  GPU1 (TUM):     PID ${PID_TUM}"
echo "  Configs: tau∈{${TAUS[*]}} × cutoff∈{${CUTOFFS[*]}}"
echo "  Results: eval_results/relpose/sensitivity/"
echo "=========================================="

wait $PID_SCANNET
STATUS_SCANNET=$?
wait $PID_TUM
STATUS_TUM=$?

echo ""
echo "=========================================="
echo "Sensitivity Sweep Complete"
echo "  ScanNet: exit code ${STATUS_SCANNET}"
echo "  TUM:     exit code ${STATUS_TUM}"
echo ""
echo "Run analysis:"
echo "  python analysis/sensitivity_analysis.py"
echo "=========================================="
