#!/bin/bash
# Re-eval ScanNet 90f with corrected first-90 protocol
set -e
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=src
export PYTHONUNBUFFERED=1

WEIGHTS="model/cut3r_512_dpt_4_64.pth"
PORT=29571
COMMON="--weights $WEIGHTS --eval_dataset scannet_s3_90 --size 512"
OUTBASE="eval_results/relpose/scannet_s3_90_first"

log() { echo "[$(date '+%H:%M:%S')] $1"; }

log "=== ScanNet 90f (first-90 protocol) re-eval ==="

for cfg in \
    "cut3r --model_update_type cut3r" \
    "ttt3r --model_update_type ttt3r" \
    "ttt3r_random --model_update_type ttt3r_random --random_gate_p 0.5" \
    "ttt3r_momentum_inv_t1 --model_update_type ttt3r_momentum --momentum_tau 1.0" \
    "ttt3r_ortho --model_update_type ttt3r_ortho --ortho_alpha_novel 0.5 --ortho_alpha_drift 0.05 --ortho_beta 0.95" \
    "ttt3r_ortho_adaptive --model_update_type ttt3r_ortho --ortho_alpha_novel 0.5 --ortho_alpha_drift 0.05 --ortho_beta 0.95 --ortho_adaptive linear"
do
    name=$(echo "$cfg" | awk '{print $1}')
    args=$(echo "$cfg" | cut -d' ' -f2-)
    log "START: $name"
    accelerate launch --num_processes 1 --main_process_port $PORT \
        eval/relpose/launch.py $COMMON $args \
        --output_dir ${OUTBASE}/${name} \
        2>&1
    log "DONE: $name"
done

log "=== All done ==="

# Print results
echo ""
echo "=== Results ==="
for d in ${OUTBASE}/*/; do
    name=$(basename "$d")
    ate=$(grep "ATE:" "$d/_error_log.txt" 2>/dev/null | awk '{gsub(/,/,""); s+=$4; n++} END {if(n>0) printf "%.4f (%d)", s/n, n}')
    echo "  $name: $ate"
done
