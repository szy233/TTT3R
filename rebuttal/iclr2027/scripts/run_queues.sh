#!/bin/bash
# Run the registered E1-E4 queues. See docs/PREREG_E1_E5.md, including Amendment 1.
# Safe to restart, since run_arm.sh skips arms that already finished.
# Run from the repository root.
ROOT=rebuttal/iclr2027/results
RUN=rebuttal/iclr2027/scripts/run_arm.sh
LOG=$ROOT/queue_all.log
OK=$ROOT/instrument_check.ok
BAD=$ROOT/instrument_check.failed
PY=${DDD3R_PYTHON:-/home/szy/anaconda3/envs/ttt3r/bin/python}
mkdir -p "$ROOT"

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
FIRST20=$(echo "$VALID" | cut -d' ' -f1-20)
[ "$(echo "$VALID" | wc -w)" = "65" ] || { echo "expected 65 valid scenes" | tee -a "$LOG"; exit 3; }

arm() { bash "$RUN" "$@" 2>&1 | tee -a "$LOG"; }

wait_for_check() {
    until [ -f "$OK" ] || [ -f "$BAD" ]; do sleep 30; done
    [ -f "$OK" ]
}

# Instrument check, as registered: TUM 1000f ddd3r within 2.2% of the logged 0.05454.
instrument_check() {
    "$PY" - "$ROOT/relpose/tum_s1_1000/ortho_check/_error_log.txt" "$OK" "$BAD" <<'PYEOF' 2>&1 | tee -a "$LOG"
import re, sys
log, ok, bad = sys.argv[1:4]
values = {}
for line in open(log, errors="ignore"):
    m = re.match(r"^(\S+?)\s*\|\s*ATE:\s*([0-9.]+)", line)
    if m:
        values[m.group(1)] = float(m.group(2))
mean = sum(values.values()) / max(len(values), 1)
gap = abs(mean - 0.05454) / 0.05454
passed = len(values) == 8 and gap <= 0.022
print(f"instrument check: n={len(values)} mean={mean:.5f} logged=0.05454 gap={100 * gap:.2f}% {'PASS' if passed else 'FAIL'}")
open(ok if passed else bad, "w").write(f"{mean:.5f}\n")
PYEOF
}

# Worker A, GPU 1: E1 and E1b.
(
    export DDD3R_PORT_OFFSET=0
    arm 1 tum_s1_1000 ortho_check
    instrument_check
    wait_for_check || exit 4
    arm 1 tum_s1_1000 normreplay
    arm 1 tum_s1_1000 normmatch
    arm 1 arkit_s1_500 ortho_check
    arm 1 arkit_s1_500 normreplay
    arm 1 arkit_s1_500 normmatch
    arm 1 scannet_s3_1000 ortho_check - $VALID
    arm 1 scannet_s3_1000 normreplay - $VALID
    arm 1 scannet_s3_1000 normmatch - $VALID
) &

# Worker B, GPU 1: Table 5 provenance, E4 and E2.
(
    export DDD3R_PORT_OFFSET=10
    wait_for_check || exit 4
    arm 1 kitti_odom brake_t2 t5_03 03
    arm 1 scannet_s3_1000 const033 - $VALID
    arm 1 tum_s1_1000 prevdir
    arm 1 scannet_s3_1000 prevdir - $VALID
) &

# Worker C, GPU 0: E3.
(
    export DDD3R_PORT_OFFSET=20
    wait_for_check || exit 4
    arm 0 tum_s1_1000 ttt3r_log
    arm 0 tum_s1_1000 ttt3r_frozen
    arm 0 tum_s1_1000 ttsa3r_log
    arm 0 tum_s1_1000 ttsa3r_frozen
    arm 0 scannet_s3_1000 ttt3r_log first20 $FIRST20
    arm 0 scannet_s3_1000 ttt3r_frozen first20 $FIRST20
    arm 0 scannet_s3_1000 ttsa3r_log first20 $FIRST20
    arm 0 scannet_s3_1000 ttsa3r_frozen first20 $FIRST20
) &

wait
echo "$(date '+%Y-%m-%d %H:%M:%S') ALL QUEUES FINISHED" | tee -a "$LOG"
