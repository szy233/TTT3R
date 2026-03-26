#!/bin/bash
# S1 Remaining: Video Depth + 7scenes (relpose already done)
set -e
export PYTHONPATH=src
GPU=${1:-1}
WEIGHTS="model/cut3r_512_dpt_4_64.pth"
PORT_BASE=29580

echo "=========================================="
echo "S1 Remaining (Video Depth + 7scenes) on GPU $GPU"
echo "=========================================="

# ── Video Depth ──
for dataset_pair in "kitti:kitti_s1_500" "bonn:bonn_s1_500" "sintel:sintel"; do
    ds_short="${dataset_pair%%:*}"
    ds_eval="${dataset_pair##*:}"
    for config in ttt3r_l2gate ttt3r_random ttt3r_conf; do
        outdir="eval_results/video_depth/${ds_eval}/$config"
        if [ -d "$outdir" ]; then
            echo "[SKIP] $ds_short/$config already exists"
            continue
        fi
        echo ""
        echo "[$(date)] Starting video_depth $ds_short: $config"
        EXTRA_ARGS=""
        if [ "$config" = "ttt3r_random" ]; then
            EXTRA_ARGS="--random_gate_p 0.5"
        fi
        CUDA_VISIBLE_DEVICES=$GPU accelerate launch --num_processes 1 --main_process_port $PORT_BASE \
            eval/video_depth/launch.py \
            --weights $WEIGHTS --output_dir $outdir \
            --eval_dataset ${ds_eval} --size 512 --model_update_type $config \
            --spectral_temperature 1.0 --geo_gate_tau 2.0 --geo_gate_freq_cutoff 4 \
            $EXTRA_ARGS
        PORT_BASE=$((PORT_BASE + 1))
        echo "[$(date)] Done video_depth $ds_short: $config"
    done
done

# ── 3D Reconstruction: 7scenes ──
for config in ttt3r_l2gate ttt3r_random ttt3r_conf; do
    outdir="eval_results/video_recon/7scenes_200/$config"
    if [ -d "$outdir" ]; then
        echo "[SKIP] 7scenes/$config already exists"
        continue
    fi
    echo ""
    echo "[$(date)] Starting 7scenes recon: $config"
    EXTRA_ARGS=""
    if [ "$config" = "ttt3r_random" ]; then
        EXTRA_ARGS="--random_gate_p 0.5"
    fi
    CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=src python eval/mv_recon/launch.py \
        --weights $WEIGHTS --output_dir $outdir \
        --eval_dataset 7scenes --size 512 --model_update_type $config \
        --spectral_temperature 1.0 --geo_gate_tau 2.0 --geo_gate_freq_cutoff 4 \
        --max_frames 200 \
        $EXTRA_ARGS
    echo "[$(date)] Done 7scenes recon: $config"
done

echo ""
echo "=========================================="
echo "[$(date)] S1 remaining evaluations complete!"
echo "=========================================="
