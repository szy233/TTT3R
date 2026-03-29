#!/bin/bash
# ScanNet scaling curve: 200f and 500f evaluations
# Run on GPU1

set -e
cd /home/szy/research/TTT3R

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=src

METHODS=(
    "cut3r"
    "ttt3r"
    "ttt3r_random"
    "ttt3r_momentum_inv_t1"
    "ttt3r_ortho"
    "ttt3r_ortho_adaptive"
)

FRAME_COUNTS=(200 500)

for NUM in "${FRAME_COUNTS[@]}"; do
    for METHOD in "${METHODS[@]}"; do
        OUTDIR="eval_results/relpose/scannet_s3_${NUM}_first/${METHOD}"
        if [ -f "${OUTDIR}/_error_log.txt" ]; then
            echo "=== SKIP ${METHOD} ${NUM}f (already done) ==="
            continue
        fi
        echo "=== Running ${METHOD} ${NUM}f ==="
        accelerate launch --num_processes 1 --main_process_port 29562 \
            eval/relpose/launch.py \
            --weights model/cut3r_512_dpt_4_64.pth \
            --output_dir "${OUTDIR}" \
            --eval_dataset "scannet_s3_${NUM}" \
            --size 512 \
            --model_update_type "${METHOD}" \
            --spectral_temperature 1.0 \
            --geo_gate_tau 2.0 \
            --geo_gate_freq_cutoff 4
        echo "=== Done ${METHOD} ${NUM}f ==="
    done
done

echo "All scaling curve evaluations complete!"
