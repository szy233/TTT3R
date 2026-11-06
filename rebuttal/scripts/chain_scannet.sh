#!/bin/bash
# Wait for the TUM TTSA3R run to finish, then run ScanNet on the same GPU.
cd /home/szy/research/TTT3R/.claude/worktrees/understand-full-project-14f204
while pgrep -f "eval_dataset tum_s1_1000 --model_update_type ttsa3r" >/dev/null; do sleep 60; done
bash rebuttal/scripts/run_ttsa3r.sh 1 scannet_s3_1000
