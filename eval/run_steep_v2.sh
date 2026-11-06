#!/bin/bash
# Steep v2 experiments: e^γ formula (paper version)
# Run sequentially on GPU1

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=src
PY=/home/szy/anaconda3/envs/ttt3r/bin/python

BASE_ARGS="--weights model/cut3r_512_dpt_4_64.pth --size 512 --model_update_type ttt3r_ortho --ortho_adaptive steep --ortho_alpha_novel 0.5 --ortho_alpha_drift 0.05 --ortho_beta 0.95"

# TUM γ=3
echo "=== TUM steep v2 γ=3 ==="
$PY -m accelerate.commands.launch --num_processes 1 --main_process_port 29575 \
    eval/relpose/launch.py $BASE_ARGS \
    --output_dir eval_results/relpose/tum_s1_1000/ttt3r_ortho_steep_v2_g3 \
    --eval_dataset tum_s1_1000 --ortho_gamma 3.0

# TUM γ=5 (larger γ needed with conservative formula)
echo "=== TUM steep v2 γ=5 ==="
$PY -m accelerate.commands.launch --num_processes 1 --main_process_port 29575 \
    eval/relpose/launch.py $BASE_ARGS \
    --output_dir eval_results/relpose/tum_s1_1000/ttt3r_ortho_steep_v2_g5 \
    --eval_dataset tum_s1_1000 --ortho_gamma 5.0

# ScanNet γ=2
echo "=== ScanNet steep v2 γ=2 ==="
$PY -m accelerate.commands.launch --num_processes 1 --main_process_port 29575 \
    eval/relpose/launch.py $BASE_ARGS \
    --output_dir eval_results/relpose/scannet_s3_1000/ttt3r_ortho_steep_v2_g2 \
    --eval_dataset scannet_s3_1000 --ortho_gamma 2.0

# ScanNet γ=3
echo "=== ScanNet steep v2 γ=3 ==="
$PY -m accelerate.commands.launch --num_processes 1 --main_process_port 29575 \
    eval/relpose/launch.py $BASE_ARGS \
    --output_dir eval_results/relpose/scannet_s3_1000/ttt3r_ortho_steep_v2_g3 \
    --eval_dataset scannet_s3_1000 --ortho_gamma 3.0

# ScanNet γ=5
echo "=== ScanNet steep v2 γ=5 ==="
$PY -m accelerate.commands.launch --num_processes 1 --main_process_port 29575 \
    eval/relpose/launch.py $BASE_ARGS \
    --output_dir eval_results/relpose/scannet_s3_1000/ttt3r_ortho_steep_v2_g5 \
    --eval_dataset scannet_s3_1000 --ortho_gamma 5.0

echo "=== All steep v2 experiments done ==="
