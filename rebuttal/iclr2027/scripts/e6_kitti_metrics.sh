#!/bin/bash
# Segment metrics for the finished E6 KITTI arms, written to results/e6_kitti_metrics.txt.
# The existing results/kitti_official_metrics.txt stays as E5b left it. Run from the repository root.
RESULTS=rebuttal/iclr2027/results
SOURCE="$PWD/$RESULTS/relpose/kitti_odo_full"
VIEW=$RESULTS/relpose/kitti_odo_full__e6_view
PY=${DDD3R_PYTHON:-/home/szy/anaconda3/envs/ttt3r/bin/python}

mkdir -p "$VIEW"
for name in ortho_check normreplay ortho_a015 const010 const015 const020 const025 const033; do
    if grep -q 'Average ATE' "$SOURCE/$name/_error_log.txt" 2>/dev/null; then
        ln -sfn "$SOURCE/$name" "$VIEW/$name"
    fi
done
"$PY" eval/relpose/eval_kitti_odom.py --results_dir "$VIEW" > "$RESULTS/e6_kitti_metrics.txt"
