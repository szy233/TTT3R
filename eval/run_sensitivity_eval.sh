#!/bin/bash
# =============================================================================
# TTT3R Hyperparameter Sensitivity Sweep — τ × freq_cutoff on Formal Eval
# =============================================================================
# Grid: τ ∈ {0.5, 1.0, 2.0, 4.0} × cutoff ∈ {2, 4, 8} = 12 configs
# Model: ttt3r_joint (L23+ttt3r, final method)
#
# GPU0: ScanNet relpose + KITTI depth
# GPU1: TUM relpose + Bonn depth + Sintel depth
#
# Usage:
#   cd /home/szy/research/TTT3R
#   conda activate ttt3r
#   nohup bash eval/run_sensitivity_eval.sh > eval/sensitivity_gpu0.log 2>&1 &
# =============================================================================
set -e
cd "$(dirname "$0")/.."

MODEL_WEIGHTS="model/cut3r_512_dpt_4_64.pth"
MODEL_TYPE="ttt3r_joint"
SPECTRAL_TAU=1.0      # Layer 2 τ fixed at best value

TAUS=("0.5" "1.0" "2.0" "4.0")
CUTOFFS=(2 4 8)

mkdir -p eval_results/sensitivity/relpose/scannet_s3_1000
mkdir -p eval_results/sensitivity/relpose/tum_s1_1000
mkdir -p eval_results/sensitivity/video_depth/kitti_s1_500
mkdir -p eval_results/sensitivity/video_depth/bonn_s1_500
mkdir -p eval_results/sensitivity/video_depth/sintel

# ─────────────────────────────────────────────────────────────────────────────
# GPU 0: ScanNet relpose + KITTI depth
# ─────────────────────────────────────────────────────────────────────────────
run_gpu0() {
    echo "[GPU0] Starting ScanNet relpose sweep (${#TAUS[@]} τ × ${#CUTOFFS[@]} cutoff = $((${#TAUS[@]} * ${#CUTOFFS[@]})) configs)"

    for tau in "${TAUS[@]}"; do
        for cutoff in "${CUTOFFS[@]}"; do
            tag="tau${tau}_c${cutoff}"
            output_dir="eval_results/sensitivity/relpose/scannet_s3_1000/${tag}"

            # Skip if already done
            if [ -f "${output_dir}/_error_log.txt" ]; then
                echo "[GPU0/ScanNet] ${tag} already done, skipping."
                continue
            fi

            echo "[GPU0/ScanNet] ${tag} => ${output_dir}"
            CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src accelerate launch \
                --num_processes 1 --main_process_port 29570 \
                eval/relpose/launch.py \
                --weights "$MODEL_WEIGHTS" \
                --output_dir "$output_dir" \
                --eval_dataset scannet_s3_1000 \
                --size 512 \
                --model_update_type "$MODEL_TYPE" \
                --spectral_temperature $SPECTRAL_TAU \
                --geo_gate_tau "$tau" \
                --geo_gate_freq_cutoff "$cutoff"
            echo "[GPU0/ScanNet] ✓ ${tag} done"
        done
    done

    echo ""
    echo "[GPU0] Starting KITTI depth sweep ..."

    for tau in "${TAUS[@]}"; do
        for cutoff in "${CUTOFFS[@]}"; do
            tag="tau${tau}_c${cutoff}"
            output_dir="eval_results/sensitivity/video_depth/kitti_s1_500/${tag}"

            if [ -f "${output_dir}/result_scale.json" ]; then
                echo "[GPU0/KITTI] ${tag} already done, skipping."
                continue
            fi

            echo "[GPU0/KITTI] ${tag} => ${output_dir}"
            CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src accelerate launch \
                --num_processes 1 --main_process_port 29570 \
                eval/video_depth/launch.py \
                --weights "$MODEL_WEIGHTS" \
                --output_dir "$output_dir" \
                --eval_dataset kitti_s1_500 \
                --size 512 \
                --model_update_type "$MODEL_TYPE" \
                --spectral_temperature $SPECTRAL_TAU \
                --geo_gate_tau "$tau" \
                --geo_gate_freq_cutoff "$cutoff"

            PYTHONPATH=src python eval/video_depth/eval_depth.py \
                --output_dir "$output_dir" \
                --eval_dataset kitti_s1_500 \
                --align scale

            echo "[GPU0/KITTI] ✓ ${tag} done"
        done
    done

    echo "[GPU0] All done."
}

# ─────────────────────────────────────────────────────────────────────────────
# GPU 1: TUM relpose + Bonn depth + Sintel depth
# ─────────────────────────────────────────────────────────────────────────────
run_gpu1() {
    echo "[GPU1] Starting TUM relpose sweep (${#TAUS[@]} τ × ${#CUTOFFS[@]} cutoff = $((${#TAUS[@]} * ${#CUTOFFS[@]})) configs)"

    for tau in "${TAUS[@]}"; do
        for cutoff in "${CUTOFFS[@]}"; do
            tag="tau${tau}_c${cutoff}"
            output_dir="eval_results/sensitivity/relpose/tum_s1_1000/${tag}"

            if [ -f "${output_dir}/_error_log.txt" ]; then
                echo "[GPU1/TUM] ${tag} already done, skipping."
                continue
            fi

            echo "[GPU1/TUM] ${tag} => ${output_dir}"
            CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src accelerate launch \
                --num_processes 1 --main_process_port 29571 \
                eval/relpose/launch.py \
                --weights "$MODEL_WEIGHTS" \
                --output_dir "$output_dir" \
                --eval_dataset tum_s1_1000 \
                --size 512 \
                --model_update_type "$MODEL_TYPE" \
                --spectral_temperature $SPECTRAL_TAU \
                --geo_gate_tau "$tau" \
                --geo_gate_freq_cutoff "$cutoff"
            echo "[GPU1/TUM] ✓ ${tag} done"
        done
    done

    echo ""
    echo "[GPU1] Starting Bonn depth sweep ..."

    for tau in "${TAUS[@]}"; do
        for cutoff in "${CUTOFFS[@]}"; do
            tag="tau${tau}_c${cutoff}"
            output_dir="eval_results/sensitivity/video_depth/bonn_s1_500/${tag}"

            if [ -f "${output_dir}/result_scale.json" ]; then
                echo "[GPU1/Bonn] ${tag} already done, skipping."
                continue
            fi

            echo "[GPU1/Bonn] ${tag} => ${output_dir}"
            CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src accelerate launch \
                --num_processes 1 --main_process_port 29571 \
                eval/video_depth/launch.py \
                --weights "$MODEL_WEIGHTS" \
                --output_dir "$output_dir" \
                --eval_dataset bonn_s1_500 \
                --size 512 \
                --model_update_type "$MODEL_TYPE" \
                --spectral_temperature $SPECTRAL_TAU \
                --geo_gate_tau "$tau" \
                --geo_gate_freq_cutoff "$cutoff"

            PYTHONPATH=src python eval/video_depth/eval_depth.py \
                --output_dir "$output_dir" \
                --eval_dataset bonn_s1_500 \
                --align scale

            echo "[GPU1/Bonn] ✓ ${tag} done"
        done
    done

    echo ""
    echo "[GPU1] Starting Sintel depth sweep ..."

    for tau in "${TAUS[@]}"; do
        for cutoff in "${CUTOFFS[@]}"; do
            tag="tau${tau}_c${cutoff}"
            output_dir="eval_results/sensitivity/video_depth/sintel/${tag}"

            if [ -f "${output_dir}/result_scale.json" ]; then
                echo "[GPU1/Sintel] ${tag} already done, skipping."
                continue
            fi

            echo "[GPU1/Sintel] ${tag} => ${output_dir}"
            CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src accelerate launch \
                --num_processes 1 --main_process_port 29571 \
                eval/video_depth/launch.py \
                --weights "$MODEL_WEIGHTS" \
                --output_dir "$output_dir" \
                --eval_dataset sintel \
                --size 512 \
                --model_update_type "$MODEL_TYPE" \
                --spectral_temperature $SPECTRAL_TAU \
                --geo_gate_tau "$tau" \
                --geo_gate_freq_cutoff "$cutoff"

            PYTHONPATH=src python eval/video_depth/eval_depth.py \
                --output_dir "$output_dir" \
                --eval_dataset sintel \
                --align scale

            echo "[GPU1/Sintel] ✓ ${tag} done"
        done
    done

    echo "[GPU1] All done."
}

# ─────────────────────────────────────────────────────────────────────────────
# Launch both GPUs in parallel
# ─────────────────────────────────────────────────────────────────────────────
echo "============================================================"
echo "TTT3R Sensitivity Sweep — τ × freq_cutoff"
echo "Grid: τ ∈ {${TAUS[*]}} × cutoff ∈ {${CUTOFFS[*]}}"
echo "Model: ${MODEL_TYPE}"
echo "============================================================"

run_gpu0 > eval/sensitivity_gpu0.log 2>&1 &
PID0=$!

run_gpu1 > eval/sensitivity_gpu1.log 2>&1 &
PID1=$!

echo "Launched:"
echo "  GPU0 (ScanNet + KITTI): PID ${PID0}  log: eval/sensitivity_gpu0.log"
echo "  GPU1 (TUM + Bonn + Sintel): PID ${PID1}  log: eval/sensitivity_gpu1.log"
echo ""
echo "Monitor: tail -f eval/sensitivity_gpu0.log eval/sensitivity_gpu1.log"
echo ""

wait $PID0; STATUS0=$?
wait $PID1; STATUS1=$?

echo ""
echo "============================================================"
echo "Sweep complete."
echo "  GPU0 exit: ${STATUS0}"
echo "  GPU1 exit: ${STATUS1}"
echo ""
echo "Next: python analysis/sensitivity_analysis.py"
echo "============================================================"
