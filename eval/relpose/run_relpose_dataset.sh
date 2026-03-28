#!/bin/bash

set -euo pipefail

workdir="${WORKDIR:-.}"
dataset="${DATASET:?DATASET is required, e.g. nuscenes_relpose or waymo_relpose}"
num_processes="${NUM_PROCESSES:-1}"
size="${SIZE:-512}"
main_port="${MAIN_PORT:-29562}"

experiments=(
  "cut3r:cut3r:0.15"
  "ttt3r:ttt3r:0.15"
  "ttt3r_momentum_inv_t1:ttt3r_momentum_inv_t1:0.15"
  "ttt3r_momentum_inv_t1_drift0:ttt3r_momentum_inv_t1:0.0"
)

ckpt_name="cut3r_512_dpt_4_64"
model_weights="${MODEL_WEIGHTS:-${workdir}/src/${ckpt_name}.pth}"
if [ ! -f "$model_weights" ]; then
  model_weights="${workdir}/model/${ckpt_name}.pth"
fi
if [ ! -f "$model_weights" ]; then
  echo "missing checkpoint: ${ckpt_name}.pth (checked src/ and model/)"
  exit 1
fi

dataset_root="${DATASET_ROOT:-${workdir}/data/${dataset}}"
if [ ! -d "$dataset_root" ]; then
  echo "missing dataset root: $dataset_root"
  exit 1
fi

for exp in "${experiments[@]}"; do
  IFS=":" read -r output_tag model_name alpha_drift <<< "$exp"
  output_dir="${workdir}/eval_results/relpose/${dataset}/${output_tag}"
  echo "[relpose] dataset=${dataset} model=${model_name} alpha_drift=${alpha_drift} -> ${output_dir}"

  accelerate launch --num_processes "$num_processes" --main_process_port "$main_port" eval/relpose/launch.py \
    --weights "$model_weights" \
    --output_dir "$output_dir" \
    --eval_dataset "$dataset" \
    --size "$size" \
    --model_update_type "$model_name" \
    --alpha_drift "$alpha_drift"
done
