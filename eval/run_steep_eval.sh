#!/bin/bash
set -e
cd /home/szy/research/TTT3R

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=src

# Steep adaptive: gamma=2 and gamma=3, on TUM 1000f and ScanNet 1000f
CONFIGS=(
    "ttt3r_ortho_steep_g2|--ortho_adaptive steep --ortho_gamma 2.0"
    "ttt3r_ortho_steep_g3|--ortho_adaptive steep --ortho_gamma 3.0"
)

DATASETS=(
    "tum_s1_1000"
    "scannet_s3_1000"
)

for DS in "${DATASETS[@]}"; do
    for CFG in "${CONFIGS[@]}"; do
        NAME="${CFG%%|*}"
        ARGS="${CFG##*|}"
        OUTDIR="eval_results/relpose/${DS}/${NAME}"
        if [ -f "${OUTDIR}/_error_log.txt" ]; then
            echo "=== SKIP ${NAME} ${DS} (already done) ==="
            continue
        fi
        echo "=== Running ${NAME} on ${DS} ==="
        python -m accelerate.commands.launch --num_processes 1 --main_process_port 29561 \
            eval/relpose/launch.py \
            --weights model/cut3r_512_dpt_4_64.pth \
            --output_dir "${OUTDIR}" \
            --eval_dataset "${DS}" \
            --size 512 \
            --model_update_type ttt3r_ortho \
            ${ARGS}
        echo "=== Done ${NAME} on ${DS} ==="
    done
done

echo "All steep evaluations complete!"
