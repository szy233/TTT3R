#!/bin/bash
# Run E9: the prefix drift energy (I1 first) and the three Sintel arms, then the ruling.
# See docs/PREREG_E9.md. Run from the repository root.
#   bash rebuttal/iclr2027/scripts/run_e9.sh <GPU_PREFIX> <GPU_SINTEL>
GPU_P=${1:-1}
GPU_S=${2:-0}
ROOT=rebuttal/iclr2027/results
RUN=rebuttal/iclr2027/scripts/run_arm.sh
LOG=${DDD3R_QUEUE_LOG:-$ROOT/queue_e9.log}
PY=${DDD3R_PYTHON:-/home/szy/anaconda3/envs/ttt3r/bin/python}
REGISTRATION=rebuttal/iclr2027/docs/PREREG_E9.md

# The registered text ends at the first line holding only "---".
want=$(cut -d' ' -f1 "$ROOT/e9_registration.sha256")
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

sha256sum rebuttal/iclr2027/scripts/e9_selector.py | tee -a "$LOG"
stamp() { date '+%Y-%m-%d %H:%M:%S'; }

# Worker A: I1, then the held-out prefix drift energy. e9_selector.py exits 4 on an I1 miss.
(
    echo "$(stamp) START prefix gpu$GPU_P" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=$GPU_P PYTHONPATH=src "$PY" rebuttal/iclr2027/scripts/e9_selector.py prefix \
        > "$ROOT/logs/e9_prefix.log" 2>&1
    echo "$(stamp) DONE  prefix exit=$?" | tee -a "$LOG"
) &

# Worker B: the Sintel arms.
(
    export DDD3R_PORT_OFFSET=60
    for a in ttt3r_plain ortho_plain brake_t1; do
        bash "$RUN" "$GPU_S" sintel "$a" 2>&1 | tee -a "$LOG"
    done
) &

wait
/usr/bin/grep -q "I1 PASS" "$ROOT/e9_instrument.txt" 2>/dev/null || { echo "I1 did not pass. No ruling." | tee -a "$LOG"; exit 4; }
"$PY" rebuttal/iclr2027/scripts/e9_selector.py rule > "$ROOT/logs/e9_rule.log" 2>&1
echo "$(stamp) E9 FINISHED rule exit=$?" | tee -a "$LOG"
