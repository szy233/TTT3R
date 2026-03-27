#!/bin/bash
# 等 l2gate_fixed 跑完后启动过夜实验
# 用法: nohup bash eval/start_overnight.sh > eval/overnight.log 2>&1 &

cd /home/szy/research/TTT3R

echo "[$(date)] Waiting for l2gate_fixed (PID 3096412) to finish..."
while kill -0 3096412 2>/dev/null; do
    sleep 60
done
echo "[$(date)] l2gate_fixed finished. Starting overnight experiments..."

bash eval/overnight_gpu0.sh
