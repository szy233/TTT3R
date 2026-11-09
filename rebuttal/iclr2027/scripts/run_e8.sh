#!/bin/bash
# Run the registered E8b queues: ARKitScenes at 1000 frames on 8 scenes. See docs/PREREG_E8.md.
# Safe to restart, since run_arm.sh skips arms that already finished.
# Run from the repository root.
ROOT=rebuttal/iclr2027/results
RUN=rebuttal/iclr2027/scripts/run_arm.sh
LOG=${DDD3R_QUEUE_LOG:-$ROOT/queue_e8.log}
REGISTRATION=rebuttal/iclr2027/docs/PREREG_E8.md

# The registered text ends at the first line holding only "---".
want=$(cut -d' ' -f1 "$ROOT/e8_registration.sha256")
have=$(sed '/^---$/q' "$REGISTRATION" | sha256sum | cut -d' ' -f1)
if [ "$want" != "$have" ]; then
    echo "registration changed: want $want, have $have. Not starting." | tee -a "$LOG"
    exit 5
fi

# The model code must be the code E7 ran. /usr/bin/git avoids shell wrappers that rewrite git output.
want=$(cut -d' ' -f1 "$ROOT/e7_code.sha256")
have=$(/usr/bin/git diff 43f7e29 -- src eval | sha256sum | cut -d' ' -f1)
if [ "$want" != "$have" ]; then
    echo "code changed since E7: want $want, have $have. Not starting." | tee -a "$LOG"
    exit 6
fi

ls src/croco/models/curope/*.so > /dev/null 2>&1 || { echo "compiled RoPE missing" | tee -a "$LOG"; exit 7; }

SCENES="41069021 41069025 41069046 41125696 41125700 41125709 41125718 41125722"
for s in $SCENES; do
    [ -d "data/long_arkit_s1/$s/rgb_1000" ] || { echo "missing data/long_arkit_s1/$s/rgb_1000" | tee -a "$LOG"; exit 3; }
done

arm() {
    bash "$RUN" "$1" arkit_s1_1000 "$2" eight $SCENES 2>&1 | tee -a "$LOG"
}

# Worker A, GPU 0: normreplay needs the norms that ortho_check logs, so they run in order.
(
    export DDD3R_PORT_OFFSET=40
    for a in ortho_check normreplay orthop ttt3r_plain; do arm 0 "$a"; done
) &

# Worker B, GPU 1.
(
    export DDD3R_PORT_OFFSET=50
    for a in dir033 const033 dir010 const010 brake_t1; do arm 1 "$a"; done
) &

wait
echo "$(date '+%Y-%m-%d %H:%M:%S') E8 QUEUES FINISHED" | tee -a "$LOG"
