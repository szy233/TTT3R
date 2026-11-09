#!/bin/bash
# Segment metrics for the finished E7 KITTI arms, written to results/e7_kitti_metrics.txt.
# The E5b and E6 metric files stay as they are. Run from the repository root.
RESULTS=rebuttal/iclr2027/results
SOURCE="$PWD/$RESULTS/relpose/kitti_odo_full"
VIEW=$RESULTS/relpose/kitti_odo_full__e7_view
PY=${DDD3R_PYTHON:-/home/szy/anaconda3/envs/ttt3r/bin/python}

mkdir -p "$VIEW"
for name in dir010 dir020 dir033 orthop const033p; do
    if grep -q 'Average ATE' "$SOURCE/$name/_error_log.txt" 2>/dev/null; then
        ln -sfn "$SOURCE/$name" "$VIEW/$name"
    fi
done
"$PY" eval/relpose/eval_kitti_odom.py --results_dir "$VIEW" > "$RESULTS/e7_kitti_metrics.txt"
