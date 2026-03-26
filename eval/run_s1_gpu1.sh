#!/bin/bash
# S1 on GPU1: Bonn video depth + 7scenes ttt3r_random + ttt3r_conf
set -e
export PYTHONPATH=src
GPU=1
WEIGHTS="model/cut3r_512_dpt_4_64.pth"
PORT_BASE=29590

# ── Bonn ──
for config in ttt3r_l2gate ttt3r_random ttt3r_conf; do
    echo "[$(date)] video_depth bonn: $config"
    EXTRA_ARGS=""; [ "$config" = "ttt3r_random" ] && EXTRA_ARGS="--random_gate_p 0.5"
    CUDA_VISIBLE_DEVICES=$GPU accelerate launch --num_processes 1 --main_process_port $PORT_BASE \
        eval/video_depth/launch.py \
        --weights $WEIGHTS --output_dir eval_results/video_depth/bonn_s1_500/$config \
        --eval_dataset bonn_s1_500 --size 512 --model_update_type $config \
        --spectral_temperature 1.0 --geo_gate_tau 2.0 --geo_gate_freq_cutoff 4 $EXTRA_ARGS
    PORT_BASE=$((PORT_BASE + 1))
done

# ── 7scenes: ttt3r_random + ttt3r_conf ──
for config in ttt3r_random ttt3r_conf; do
    echo "[$(date)] 7scenes: $config"
    EXTRA_ARGS=""; [ "$config" = "ttt3r_random" ] && EXTRA_ARGS="--random_gate_p 0.5"
    CUDA_VISIBLE_DEVICES=$GPU python eval/mv_recon/launch.py \
        --weights $WEIGHTS --output_dir eval_results/video_recon/7scenes_200/$config \
        --eval_dataset 7scenes --size 512 --model_update_type $config \
        --spectral_temperature 1.0 --geo_gate_tau 2.0 --geo_gate_freq_cutoff 4 --max_frames 200 \
        $EXTRA_ARGS
done

echo "[$(date)] GPU1 done!"
