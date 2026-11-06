#!/bin/bash
# E5b: all five methods on KITTI Odometry with compiled RoPE. See docs/PREREG_E1_E5.md,
# Amendment 2. Safe to restart, since finished sequences and arms are skipped.
# Run from the repository root, after run_queues.sh has started.
ROOT=rebuttal/iclr2027/results
RUN=rebuttal/iclr2027/scripts/run_arm.sh
LOG=$ROOT/queue_all.log
export DDD3R_PORT_OFFSET=50

# Same gate as the other workers: no ruling on a code base that fails the instrument check.
until [ -f "$ROOT/instrument_check.ok" ] || [ -f "$ROOT/instrument_check.failed" ]; do sleep 30; done
[ -f "$ROOT/instrument_check.ok" ] || { echo "E5b not started: instrument check failed" | tee -a "$LOG"; exit 4; }

arm() { bash "$RUN" "$@" 2>&1 | tee -a "$LOG"; }

# Does the earlier TTSA3R KITTI run still reproduce? Decides whether its row can be filled.
arm 0 kitti_odo_full ttsa3r_log ttsacheck04 04

for method in ortho_plain brake_t1 const05 ttt3r_plain cut3r; do
    arm 0 kitti_odo_full "$method"
done
echo "$(date '+%Y-%m-%d %H:%M:%S') E5b FINISHED" | tee -a "$LOG"
