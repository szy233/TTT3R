#!/bin/bash
# Run the registered E6d queues. See docs/PREREG_E6D.md.
# Safe to restart, since run_arm.sh skips arms that already finished.
# Run from the repository root.
ROOT=rebuttal/iclr2027/results
RUN=rebuttal/iclr2027/scripts/run_arm.sh
LOG=${DDD3R_QUEUE_LOG:-$ROOT/queue_e6d.log}
REGISTRATION=rebuttal/iclr2027/docs/PREREG_E6D.md

# The registered text ends at the first line holding only "---". Dated results may follow it.
want=$(cut -d' ' -f1 "$ROOT/e6d_registration.sha256")
have=$(sed '/^---$/q' "$REGISTRATION" | sha256sum | cut -d' ' -f1)
if [ "$want" != "$have" ]; then
    echo "registration changed: want $want, have $have. Not starting." | tee -a "$LOG"
    exit 5
fi

arm() { bash "$RUN" "$@" 2>&1 | tee -a "$LOG"; }

# Two workers on GPU 1, which has the free memory. const033 goes first since it carries the check.
(
    export DDD3R_PORT_OFFSET=50
    arm 1 kitti_odo_full const033
    arm 1 kitti_odo_full const020
    arm 1 kitti_odo_full const010
) &
(
    export DDD3R_PORT_OFFSET=60
    arm 1 kitti_odo_full const025
    arm 1 kitti_odo_full const015
) &

wait
echo "$(date '+%Y-%m-%d %H:%M:%S') E6D QUEUES FINISHED" | tee -a "$LOG"
