#!/bin/bash
# Run the registered E6 queues. See docs/PREREG_E6.md.
# Safe to restart, since run_arm.sh skips arms that already finished.
# Run from the repository root.
ROOT=rebuttal/iclr2027/results
RUN=rebuttal/iclr2027/scripts/run_arm.sh
LOG=$ROOT/queue_e6.log
REGISTRATION=rebuttal/iclr2027/docs/PREREG_E6.md
PY=${DDD3R_PYTHON:-/home/szy/anaconda3/envs/ttt3r/bin/python}

# The registered text ends at the first line holding only "---". Dated results may follow it.
want=$(cut -d' ' -f1 "$ROOT/e6_registration.sha256")
have=$(sed '/^---$/q' "$REGISTRATION" | sha256sum | cut -d' ' -f1)
if [ "$want" != "$have" ]; then
    echo "registration changed: want $want, have $have. Not starting." | tee -a "$LOG"
    exit 5
fi

VALID=$("$PY" - <<'PYEOF'
import re
names = set()
for line in open("/home/szy/research/TTT3R/eval_results/relpose/scannet_s3_1000/ttt3r/_error_log.txt", errors="ignore"):
    m = re.match(r"^(\S+?)\s*\|\s*ATE:", line)
    if m:
        names.add(m.group(1).split("-", 1)[-1])
print(" ".join(sorted(names)))
PYEOF
)
[ "$(echo "$VALID" | wc -w)" = "65" ] || { echo "expected 65 valid scenes" | tee -a "$LOG"; exit 3; }

arm() { bash "$RUN" "$@" 2>&1 | tee -a "$LOG"; }

# Worker A, GPU 1: E6a first, then the out-of-sample part of E6c.
(
    export DDD3R_PORT_OFFSET=0
    arm 1 kitti_odo_full ortho_check
    arm 1 kitti_odo_full normreplay
    arm 1 arkit_s1_500 ortho_a015
    arm 1 arkit_s1_500 normreplay_a015
    arm 1 kitti_odo_full ortho_a015
) &

# Worker B, GPU 0: E6b first, then the TUM part of E6c.
(
    export DDD3R_PORT_OFFSET=10
    for flat in const010 const015 const020 const025 const033; do
        arm 0 tum_s1_1000 "$flat"
    done
    arm 0 tum_s1_1000 ortho_a015
    arm 0 tum_s1_1000 normreplay_a015
) &

# Worker C, GPU 1: the ScanNet part of E6c, the longest arms.
(
    export DDD3R_PORT_OFFSET=20
    arm 1 scannet_s3_1000 ortho_a015 - $VALID
    arm 1 scannet_s3_1000 normreplay_a015 - $VALID
) &

wait
echo "$(date '+%Y-%m-%d %H:%M:%S') E6 QUEUES FINISHED" | tee -a "$LOG"
