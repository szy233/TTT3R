#!/bin/bash
# =============================================================================
# DDD3R Unified Evaluation Script (paper naming)
# Usage: bash eval/run_ddd3r_eval.sh [GPU_ID] [DATASET] [METHOD]
#
# Methods: cut3r, ttt3r, ddd3r_constant, ddd3r_constant_p{N}, ddd3r_brake, ddd3r, ddd3r_g{N}
#   ddd3r_constant      → alpha=0.5 (ScanNet default)
#   ddd3r_constant_p33  → alpha=0.33 (TUM best), ddd3r_constant_p5 → alpha=0.5
#   ddd3r_g2            → gamma=2, ddd3r_g0.5 → gamma=0.5
#
# Datasets:
#   Relpose:     tum_s1_1000, tum_s1_90, scannet_s3_1000, scannet_s3_90, sintel, kitti_odom
#   Video depth: kitti, bonn, sintel_depth
#   3D recon:    7scenes
#
# Video depth alignment: scale&shift (default, matches CUT3R/MonST3R protocol)
# =============================================================================

set -e

GPU=${1:-0}
DATASET=${2:-tum_s1_1000}
METHOD=${3:-ddd3r}

export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=src
PY=${DDD3R_PYTHON:-/home/szy/anaconda3/envs/ttt3r/bin/python}
WEIGHTS="model/cut3r_512_dpt_4_64.pth"
PORT_OFFSET=${DDD3R_PORT_OFFSET:-0}
PORT=$((29560 + GPU + PORT_OFFSET))

# Parse method → model_update_type + DDD3R params
case "$METHOD" in
    cut3r)
        UPDATE_TYPE="cut3r"
        EXTRA_ARGS=""
        ;;
    ttt3r)
        UPDATE_TYPE="ttt3r"
        EXTRA_ARGS=""
        ;;
    ddd3r_constant)
        UPDATE_TYPE="ddd3r_constant"
        EXTRA_ARGS="--alpha 0.5"
        ;;
    ddd3r_constant_p*)
        # e.g. ddd3r_constant_p33 → alpha=0.33, ddd3r_constant_p05 → alpha=0.05
        P=$(echo "$METHOD" | sed 's/ddd3r_constant_p//' | sed 's/^/0./')
        UPDATE_TYPE="ddd3r_constant"
        EXTRA_ARGS="--alpha $P"
        ;;
    ddd3r_brake)
        UPDATE_TYPE="ddd3r_brake"
        EXTRA_ARGS="--brake_tau 1.0"
        ;;
    ddd3r)
        UPDATE_TYPE="ddd3r"
        EXTRA_ARGS="--alpha_perp 0.5 --alpha_parallel 0.05 --beta_ema 0.95 --gamma 0"
        ;;
    ddd3r_g*)
        # e.g. ddd3r_g2 → gamma=2, ddd3r_g0.5 → gamma=0.5
        GAMMA=$(echo "$METHOD" | sed 's/ddd3r_g//')
        UPDATE_TYPE="ddd3r"
        EXTRA_ARGS="--alpha_perp 0.5 --alpha_parallel 0.05 --beta_ema 0.95 --gamma $GAMMA"
        METHOD="ddd3r_g${GAMMA}"
        ;;
    # --- Auto-gamma methods ---
    ddd3r_auto_warmup_linear)
        UPDATE_TYPE="ddd3r"
        EXTRA_ARGS="--alpha_perp 0.5 --alpha_parallel 0.05 --beta_ema 0.95 --auto_gamma warmup_linear --auto_gamma_warmup 30 --auto_gamma_max 3.0"
        ;;
    ddd3r_auto_warmup_threshold)
        UPDATE_TYPE="ddd3r"
        EXTRA_ARGS="--alpha_perp 0.5 --alpha_parallel 0.05 --beta_ema 0.95 --auto_gamma warmup_threshold --auto_gamma_warmup 30 --auto_gamma_max 3.0"
        ;;
    ddd3r_warmup_local_de)
        UPDATE_TYPE="ddd3r"
        EXTRA_ARGS="--alpha_perp 0.5 --alpha_parallel 0.05 --beta_ema 0.95 --auto_gamma warmup_local_de --auto_gamma_warmup 30 --auto_gamma_max 3.0"
        ;;
    ddd3r_warmup_local_de_linear)
        UPDATE_TYPE="ddd3r"
        EXTRA_ARGS="--alpha_perp 0.5 --alpha_parallel 0.05 --beta_ema 0.95 --auto_gamma warmup_local_de_linear --auto_gamma_warmup 30 --auto_gamma_max 3.0"
        ;;
    ddd3r_auto_steep_sigmoid)
        UPDATE_TYPE="ddd3r"
        EXTRA_ARGS="--alpha_perp 0.5 --alpha_parallel 0.05 --beta_ema 0.95 --auto_gamma steep_sigmoid --auto_gamma_k 10.0"
        ;;
    ddd3r_auto_steep_sigmoid_k20)
        UPDATE_TYPE="ddd3r"
        EXTRA_ARGS="--alpha_perp 0.5 --alpha_parallel 0.05 --beta_ema 0.95 --auto_gamma steep_sigmoid --auto_gamma_k 20.0"
        ;;
    ddd3r_auto_steep_clamp)
        UPDATE_TYPE="ddd3r"
        EXTRA_ARGS="--alpha_perp 0.5 --alpha_parallel 0.05 --beta_ema 0.95 --auto_gamma steep_clamp --auto_gamma_lo 0.3 --auto_gamma_hi 0.6"
        ;;
    ddd3r_auto_steep_clamp_tight)
        UPDATE_TYPE="ddd3r"
        EXTRA_ARGS="--alpha_perp 0.5 --alpha_parallel 0.05 --beta_ema 0.95 --auto_gamma steep_clamp --auto_gamma_lo 0.35 --auto_gamma_hi 0.55"
        ;;
    # --- Attention entropy adaptive ---
    ddd3r_entropy)
        UPDATE_TYPE="ddd3r"
        EXTRA_ARGS="--alpha_perp 0.5 --alpha_parallel 0.05 --beta_ema 0.95 --auto_gamma entropy"
        ;;
    ddd3r_de)
        UPDATE_TYPE="ddd3r"
        EXTRA_ARGS="--alpha_perp 0.5 --alpha_parallel 0.05 --beta_ema 0.95 --auto_gamma drift_energy"
        ;;
    ddd3r_deg)
        UPDATE_TYPE="ddd3r"
        EXTRA_ARGS="--alpha_perp 0.5 --alpha_parallel 0.05 --beta_ema 0.95 --auto_gamma drift_energy_gate --de_gate_threshold 0.7"
        ;;
    ddd3r_deg_t*)
        # e.g. ddd3r_deg_t5 → threshold=0.5, ddd3r_deg_t8 → 0.8
        THRESH=$(echo "$METHOD" | sed 's/ddd3r_deg_t//' | sed 's/^/0./')
        UPDATE_TYPE="ddd3r"
        EXTRA_ARGS="--alpha_perp 0.5 --alpha_parallel 0.05 --beta_ema 0.95 --auto_gamma drift_energy_gate --de_gate_threshold $THRESH"
        ;;
    ddd3r_entropy_b*)
        # e.g. ddd3r_entropy_b9 → entropy_ema_beta=0.9, ddd3r_entropy_b99 → 0.99
        EBETA=$(echo "$METHOD" | sed 's/ddd3r_entropy_b//' | sed 's/^/0./')
        UPDATE_TYPE="ddd3r"
        EXTRA_ARGS="--alpha_perp 0.5 --alpha_parallel 0.05 --beta_ema 0.95 --auto_gamma entropy --entropy_ema_beta $EBETA"
        ;;
    # --- Local drift energy adaptive ---
    ddd3r_local_de)
        UPDATE_TYPE="ddd3r"
        EXTRA_ARGS="--alpha_perp 0.5 --alpha_parallel 0.05 --beta_ema 0.95 --auto_gamma local_de"
        ;;
    ddd3r_local_de_raw)
        UPDATE_TYPE="ddd3r"
        EXTRA_ARGS="--alpha_perp 0.5 --alpha_parallel 0.05 --beta_ema 0.95 --auto_gamma local_de_raw"
        ;;
    ddd3r_local_de_raw_p*)
        # e.g. ddd3r_local_de_raw_p2 → power=2, ddd3r_local_de_raw_p3 → power=3
        P=$(echo "$METHOD" | sed 's/ddd3r_local_de_raw_p//')
        UPDATE_TYPE="ddd3r"
        EXTRA_ARGS="--alpha_perp 0.5 --alpha_parallel 0.05 --beta_ema 0.95 --auto_gamma local_de_raw_p${P}"
        ;;
    ddd3r_local_de_raw_cap*)
        # e.g. ddd3r_local_de_raw_cap06 → cap_ratio=0.6
        CAP=$(echo "$METHOD" | sed 's/ddd3r_local_de_raw_cap//' | sed 's/^/0./')
        UPDATE_TYPE="ddd3r"
        EXTRA_ARGS="--alpha_perp 0.5 --alpha_parallel 0.05 --beta_ema 0.95 --auto_gamma local_de_raw_cap${CAP}"
        ;;
    ddd3r_local_de_sig)
        UPDATE_TYPE="ddd3r"
        EXTRA_ARGS="--alpha_perp 0.5 --alpha_parallel 0.05 --beta_ema 0.95 --auto_gamma local_de_sigmoid --auto_gamma_k 20.0"
        ;;
    ddd3r_local_de_sig_k*)
        # e.g. ddd3r_local_de_sig_k10 → k=10
        K=$(echo "$METHOD" | sed 's/ddd3r_local_de_sig_k//')
        UPDATE_TYPE="ddd3r"
        EXTRA_ARGS="--alpha_perp 0.5 --alpha_parallel 0.05 --beta_ema 0.95 --auto_gamma local_de_sigmoid --auto_gamma_k ${K}.0"
        ;;
    # --- Per-token local drift energy adaptive (A: linear, B: sigmoid) ---
    ddd3r_local_de_token)
        UPDATE_TYPE="ddd3r"
        EXTRA_ARGS="--alpha_perp 0.5 --alpha_parallel 0.05 --beta_ema 0.95 --auto_gamma local_de_token"
        ;;
    ddd3r_local_de_token_sig)
        UPDATE_TYPE="ddd3r"
        EXTRA_ARGS="--alpha_perp 0.5 --alpha_parallel 0.05 --beta_ema 0.95 --auto_gamma local_de_token_sig --auto_gamma_k 20.0"
        ;;
    ddd3r_local_de_token_sig_k*)
        K=$(echo "$METHOD" | sed 's/ddd3r_local_de_token_sig_k//')
        UPDATE_TYPE="ddd3r"
        EXTRA_ARGS="--alpha_perp 0.5 --alpha_parallel 0.05 --beta_ema 0.95 --auto_gamma local_de_token_sig --auto_gamma_k ${K}.0"
        ;;
    # --- Drift growth rate (G) ---
    ddd3r_drift_growth)
        UPDATE_TYPE="ddd3r"
        EXTRA_ARGS="--alpha_perp 0.5 --alpha_parallel 0.05 --beta_ema 0.95 --auto_gamma drift_growth"
        ;;
    # --- Projection fraction (H) ---
    ddd3r_proj_frac)
        UPDATE_TYPE="ddd3r"
        EXTRA_ARGS="--alpha_perp 0.5 --alpha_parallel 0.05 --beta_ema 0.95 --auto_gamma proj_frac"
        ;;
    # --- Frame-mean local DE regime switch (F: sigmoid, F2: linear) ---
    ddd3r_fmean_sig)
        UPDATE_TYPE="ddd3r"
        EXTRA_ARGS="--alpha_perp 0.5 --alpha_parallel 0.05 --beta_ema 0.95 --auto_gamma local_de_fmean_sig --auto_gamma_k 20.0 --auto_gamma_tau 0.5"
        ;;
    ddd3r_fmean_sig_k*)
        K=$(echo "$METHOD" | sed 's/ddd3r_fmean_sig_k//')
        UPDATE_TYPE="ddd3r"
        EXTRA_ARGS="--alpha_perp 0.5 --alpha_parallel 0.05 --beta_ema 0.95 --auto_gamma local_de_fmean_sig --auto_gamma_k ${K}.0 --auto_gamma_tau 0.5"
        ;;
    ddd3r_fmean_sig_t*)
        # e.g. ddd3r_fmean_sig_t45 → tau=0.45
        TAU=$(echo "$METHOD" | sed 's/ddd3r_fmean_sig_t//' | sed 's/^/0./')
        UPDATE_TYPE="ddd3r"
        EXTRA_ARGS="--alpha_perp 0.5 --alpha_parallel 0.05 --beta_ema 0.95 --auto_gamma local_de_fmean_sig --auto_gamma_k 20.0 --auto_gamma_tau $TAU"
        ;;
    ddd3r_fmean)
        UPDATE_TYPE="ddd3r"
        EXTRA_ARGS="--alpha_perp 0.5 --alpha_parallel 0.05 --beta_ema 0.95 --auto_gamma local_de_fmean"
        ;;
    # --- Momentum decomposition (I): unnormalized EMA resultant length ---
    ddd3r_momentum)
        UPDATE_TYPE="ddd3r"
        EXTRA_ARGS="--alpha_perp 0.5 --alpha_parallel 0.05 --beta_ema 0.95 --auto_gamma momentum_R"
        ;;
    # --- Novel boost (J): boost novel instead of suppressing drift ---
    # α∥ = 0.5-γ, α⊥ = 0.5. e.g. boost02 → γ=0.02, α∥=0.48
    ddd3r_boost*)
        GAMMA=$(echo "$METHOD" | sed 's/ddd3r_boost//' | sed 's/^/0./')
        ALPHA_BASE=$(python3 -c "print(0.5 - $GAMMA)")
        UPDATE_TYPE="ddd3r"
        EXTRA_ARGS="--alpha_perp 0.5 --alpha_parallel $ALPHA_BASE --beta_ema 0.95 --gamma 0"
        ;;
    # --- Ortho + Brake combination ---
    ddd3r_ortho_brake)
        UPDATE_TYPE="ddd3r_ortho_brake"
        EXTRA_ARGS="--alpha_perp 0.5 --alpha_parallel 0.05 --beta_ema 0.95 --gamma 0 --brake_tau 1.0"
        ;;
    # --- Drift direction confidence gate (J2) ---
    ddd3r_drift_conf)
        UPDATE_TYPE="ddd3r"
        EXTRA_ARGS="--alpha_perp 0.5 --alpha_parallel 0.05 --beta_ema 0.95 --auto_gamma drift_conf"
        ;;
    ddd3r_drift_conf_token)
        UPDATE_TYPE="ddd3r"
        EXTRA_ARGS="--alpha_perp 0.5 --alpha_parallel 0.05 --beta_ema 0.95 --auto_gamma drift_conf_token"
        ;;
    ddd3r_drift_conf_fallback)
        UPDATE_TYPE="ddd3r"
        EXTRA_ARGS="--alpha_perp 0.5 --alpha_parallel 0.05 --beta_ema 0.95 --auto_gamma drift_conf_fallback"
        ;;
    # --- α∅ ablation (K): vary suppression ratio --- (MUST be after ddd3r_auto* patterns)
    ddd3r_a[0-9]*)
        AP=$(echo "$METHOD" | sed 's/ddd3r_a//' | sed 's/^/0./')
        UPDATE_TYPE="ddd3r"
        EXTRA_ARGS="--alpha_perp 0.5 --alpha_parallel $AP --beta_ema 0.95 --gamma 0"
        ;;
    *)
        echo "Unknown method: $METHOD"
        echo "Available: cut3r, ttt3r, ddd3r_constant, ddd3r_constant_p{N}, ddd3r_brake, ddd3r, ddd3r_g{N}"
        echo "  Auto-gamma: ddd3r_auto_warmup_linear, ddd3r_auto_warmup_threshold"
        echo "              ddd3r_auto_steep_sigmoid, ddd3r_auto_steep_sigmoid_k20"
        echo "              ddd3r_auto_steep_clamp, ddd3r_auto_steep_clamp_tight"
        echo "  Entropy:    ddd3r_entropy, ddd3r_entropy_b{N}"
        exit 1
        ;;
esac

# Parse dataset → eval task type + actual dataset key for launch.py
EVAL_DATASET="$DATASET"
case "$DATASET" in
    tum|tum_*|scannet_*|sintel|kitti_odom)
        TASK="relpose"
        LAUNCH="eval/relpose/launch.py"
        ;;
    kitti|kitti_s1_*|bonn|bonn_s1_*|sintel_depth)
        TASK="video_depth"
        LAUNCH="eval/video_depth/launch.py"
        # sintel_depth → launch.py needs "sintel" as eval_dataset key
        if [ "$DATASET" = "sintel_depth" ]; then
            EVAL_DATASET="sintel"
        fi
        ;;
    7scenes)
        TASK="mv_recon"
        LAUNCH="eval/mv_recon/launch.py"
        ;;
    *)
        echo "Unknown dataset: $DATASET"
        exit 1
        ;;
esac

OUTPUT_DIR="eval_results/${TASK}/${DATASET}/${METHOD}"

echo "=== DDD3R Eval: ${METHOD} on ${DATASET} (GPU ${GPU}) ==="
echo "  update_type: ${UPDATE_TYPE}"
echo "  output_dir:  ${OUTPUT_DIR}"

# Step 1: Run inference
$PY -m accelerate.commands.launch --num_processes 1 --main_process_port $PORT \
    $LAUNCH \
    --weights $WEIGHTS --size 512 \
    --output_dir $OUTPUT_DIR \
    --eval_dataset $EVAL_DATASET \
    --model_update_type $UPDATE_TYPE \
    $EXTRA_ARGS

# Step 2: For video depth, run depth metric evaluation (abs_rel etc.)
if [ "$TASK" = "video_depth" ]; then
    ALIGN=${DDD3R_DEPTH_ALIGN:-scale&shift}
    echo "=== Running depth evaluation (align=${ALIGN}) ==="
    $PY eval/video_depth/eval_depth.py \
        --output_dir $OUTPUT_DIR \
        --eval_dataset $EVAL_DATASET \
        --align "$ALIGN"
fi

echo "=== Done: ${METHOD} on ${DATASET} ==="
