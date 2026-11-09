#!/bin/bash
# Run the rounding-perturbation arms of docs/PREREG_E7.md, Amendment 1.
# Safe to restart, since run_arm.sh skips arms that already finished.
# Run from the repository root.
ROOT=rebuttal/iclr2027/results
RUN=rebuttal/iclr2027/scripts/run_arm.sh
LOG=${DDD3R_QUEUE_LOG:-$ROOT/queue_e7_noise.log}
REGISTRATION=rebuttal/iclr2027/docs/PREREG_E7.md

want=$(cut -d' ' -f1 "$ROOT/e7_registration.sha256")
have=$(sed '/^---$/q' "$REGISTRATION" | sha256sum | cut -d' ' -f1)
if [ "$want" != "$have" ]; then
    echo "registration changed: want $want, have $have. Not starting." | tee -a "$LOG"
    exit 5
fi

arm() {
    [ -n "${DDD3R_ECHO_ARMS:-}" ] && echo "ARM $2 $3"
    bash "$RUN" "$@" 2>&1 | tee -a "$LOG"
}

(
    export DDD3R_PORT_OFFSET=40
    for a in orthop const033p; do arm 0 tum_s1_1000 "$a"; done
) &
(
    export DDD3R_PORT_OFFSET=50
    for a in orthop const033p; do arm 1 kitti_odo_full "$a"; done
) &

wait
echo "$(date '+%Y-%m-%d %H:%M:%S') E7 NOISE FINISHED" | tee -a "$LOG"
