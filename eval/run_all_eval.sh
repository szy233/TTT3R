#!/bin/bash
# =============================================================================
# TTT3R 全量正式评测 — 双卡并行
# 对比三个配置: cut3r (baseline), ttt3r, ttt3r_joint (L23+ttt3r)
# =============================================================================
# 任务分配:
#   GPU0: Sintel relpose → KITTI depth → Bonn depth → 7scenes recon
#   GPU1: Sintel depth   → (空闲后帮跑剩余)
# =============================================================================
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=src

MODEL_WEIGHTS="model/cut3r_512_dpt_4_64.pth"
CONFIGS=("cut3r" "ttt3r" "ttt3r_joint")

# 频域超参 (仅 ttt3r_joint 使用，传给其他配置无副作用)
SPECTRAL_TAU=1.0
GEO_TAU=2.0
GEO_CUTOFF=4

COMMON_ARGS="--weights $MODEL_WEIGHTS --size 512 \
    --spectral_temperature $SPECTRAL_TAU \
    --geo_gate_tau $GEO_TAU \
    --geo_gate_freq_cutoff $GEO_CUTOFF"

mkdir -p eval_results/relpose eval_results/video_depth eval_results/video_recon

# ── GPU 0 任务 ──
run_gpu0() {
    # 1. Sintel relpose
    echo "[GPU0] === Sintel Relpose ==="
    for config in "${CONFIGS[@]}"; do
        output_dir="eval_results/relpose/sintel/${config}"
        echo "[GPU0] relpose/sintel/${config}"
        CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 --main_process_port 29560 \
            eval/relpose/launch.py \
            $COMMON_ARGS \
            --output_dir "$output_dir" \
            --eval_dataset sintel \
            --model_update_type "$config"
    done

    # 2. KITTI video depth
    echo "[GPU0] === KITTI Video Depth ==="
    for config in "${CONFIGS[@]}"; do
        output_dir="eval_results/video_depth/kitti_s1_500/${config}"
        echo "[GPU0] video_depth/kitti/${config}"
        CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 --main_process_port 29560 \
            eval/video_depth/launch.py \
            $COMMON_ARGS \
            --output_dir "$output_dir" \
            --eval_dataset kitti_s1_500 \
            --model_update_type "$config"

        for align in metric scale "scale&shift"; do
            python eval/video_depth/eval_depth.py \
                --output_dir "$output_dir" \
                --eval_dataset kitti_s1_500 \
                --align "$align"
        done
    done

    # 3. Bonn video depth
    echo "[GPU0] === Bonn Video Depth ==="
    for config in "${CONFIGS[@]}"; do
        output_dir="eval_results/video_depth/bonn_s1_500/${config}"
        echo "[GPU0] video_depth/bonn/${config}"
        CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 --main_process_port 29560 \
            eval/video_depth/launch.py \
            $COMMON_ARGS \
            --output_dir "$output_dir" \
            --eval_dataset bonn_s1_500 \
            --model_update_type "$config"

        for align in metric scale "scale&shift"; do
            python eval/video_depth/eval_depth.py \
                --output_dir "$output_dir" \
                --eval_dataset bonn_s1_500 \
                --align "$align"
        done
    done

    echo "[GPU0] All tasks done."
}

# ── GPU 1 任务 ──
run_gpu1() {
    # 1. Sintel video depth
    echo "[GPU1] === Sintel Video Depth ==="
    for config in "${CONFIGS[@]}"; do
        output_dir="eval_results/video_depth/sintel/${config}"
        echo "[GPU1] video_depth/sintel/${config}"
        CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes 1 --main_process_port 29561 \
            eval/video_depth/launch.py \
            $COMMON_ARGS \
            --output_dir "$output_dir" \
            --eval_dataset sintel \
            --model_update_type "$config"

        for align in metric scale "scale&shift"; do
            python eval/video_depth/eval_depth.py \
                --output_dir "$output_dir" \
                --eval_dataset sintel \
                --align "$align"
        done
    done

    # 2. 7scenes 3D reconstruction
    echo "[GPU1] === 7scenes 3D Reconstruction ==="
    for config in "${CONFIGS[@]}"; do
        output_dir="eval_results/video_recon/7scenes_200/${config}"
        echo "[GPU1] video_recon/7scenes/${config}"
        CUDA_VISIBLE_DEVICES=1 NCCL_TIMEOUT=360000 accelerate launch --num_processes 1 --main_process_port 29561 \
            eval/mv_recon/launch.py \
            $COMMON_ARGS \
            --output_dir "$output_dir" \
            --model_name "$config" \
            --model_update_type "$config" \
            --max_frames 200
    done

    echo "[GPU1] All tasks done."
}

# ── 并行启动 ──
run_gpu0 &
PID_GPU0=$!

run_gpu1 &
PID_GPU1=$!

echo "=========================================="
echo "Full evaluation launched:"
echo "  GPU0: Sintel relpose → KITTI depth → Bonn depth   (PID ${PID_GPU0})"
echo "  GPU1: Sintel depth → 7scenes recon                (PID ${PID_GPU1})"
echo "  Configs: ${CONFIGS[*]}"
echo "=========================================="

wait $PID_GPU0
S0=$?
wait $PID_GPU1
S1=$?

echo ""
echo "=========================================="
echo "Evaluation Complete"
echo "  GPU0 exit: ${S0}"
echo "  GPU1 exit: ${S1}"
echo "=========================================="
echo ""
echo "Results:"
for config in "${CONFIGS[@]}"; do
    echo "  relpose/sintel/${config}"
    echo "  video_depth/kitti_s1_500/${config}"
    echo "  video_depth/bonn_s1_500/${config}"
    echo "  video_depth/sintel/${config}"
    echo "  video_recon/7scenes_200/${config}"
done
