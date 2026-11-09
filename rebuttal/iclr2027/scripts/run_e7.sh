#!/bin/bash
# Run the registered E7 queues. See docs/PREREG_E7.md.
# Safe to restart, since run_arm.sh skips arms that already finished.
# Run from the repository root.
ROOT=rebuttal/iclr2027/results
RUN=rebuttal/iclr2027/scripts/run_arm.sh
LOG=${DDD3R_QUEUE_LOG:-$ROOT/queue_e7.log}
REGISTRATION=rebuttal/iclr2027/docs/PREREG_E7.md
PY=${DDD3R_PYTHON:-/home/szy/anaconda3/envs/ttt3r/bin/python}

# The registered text ends at the first line holding only "---". Dated results may follow it.
want=$(cut -d' ' -f1 "$ROOT/e7_registration.sha256")
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

arm() {
    [ -n "${DDD3R_ECHO_ARMS:-}" ] && echo "ARM $2 $3"
    bash "$RUN" "$@" 2>&1 | tee -a "$LOG"
}

# Worker A, GPU 1: KITTI Odometry.
(
    export DDD3R_PORT_OFFSET=0
    for a in dir033 dir020 dir010; do arm 1 kitti_odo_full "$a"; done
) &

# Worker B, GPU 0: TUM first, then ARKitScenes.
(
    export DDD3R_PORT_OFFSET=10
    for a in dir033 dir020 dir010; do arm 0 tum_s1_1000 "$a"; done
    for a in dir033 const033 dir020 const020 dir010 const010; do arm 0 arkit_s1_500 "$a"; done
) &

# Worker C, GPU 1: ScanNet, the longest arms.
(
    export DDD3R_PORT_OFFSET=20
    arm 1 scannet_s3_1000 dir033 - $VALID
    arm 1 scannet_s3_1000 const010 - $VALID
) &

# Worker D, GPU 0: ScanNet.
(
    export DDD3R_PORT_OFFSET=30
    arm 0 scannet_s3_1000 dir010 - $VALID
) &

wait
echo "$(date '+%Y-%m-%d %H:%M:%S') E7 QUEUES FINISHED" | tee -a "$LOG"
