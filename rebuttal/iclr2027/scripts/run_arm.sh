#!/bin/bash
# Run one arm of E1-E5. See rebuttal/iclr2027/docs/PREREG_E1_E5.md.
#
# Usage: bash rebuttal/iclr2027/scripts/run_arm.sh <GPU> <DATASET> <ARM> [OUTPUT_TAG] [SEQ ...]
#   DATASET    a key of eval/relpose/metadata.py, e.g. tum_s1_1000
#   OUTPUT_TAG suffix for the output folder when only a subset runs ("-" for none)
#   SEQ        optional sequence names; defaults to the whole dataset
#
# An arm whose folder already holds an "Average ATE" line is skipped, so a queue
# can be restarted safely.
GPU=$1
DATASET=$2
ARM=$3
TAG=${4:--}
shift 4 2>/dev/null || shift $#
SEQS=("$@")

ROOT=rebuttal/iclr2027/results
GROUP="$DATASET"
[ "$TAG" != "-" ] && GROUP="${DATASET}__${TAG}"
OUT="$ROOT/relpose/$GROUP/$ARM"
LOG="$ROOT/logs/$GROUP.$ARM.log"
GATES="$ROOT/gates/$GROUP"
GAINS="$ROOT/gains/$GROUP/$ARM"
NORMS="$ROOT/norms/$GROUP"
mkdir -p "$OUT" "$ROOT/logs"

ORTHO="--alpha_perp 0.5 --alpha_parallel 0.05 --beta_ema 0.95 --gamma 0"
case "$ARM" in
    ortho_plain)            TYPE=ddd3r;          EXTRA="$ORTHO" ;;
    ortho_check)            TYPE=ddd3r;          EXTRA="$ORTHO --gain_log_dir $GAINS --norm_log_dir $NORMS" ;;
    normreplay)             TYPE=ddd3r;          EXTRA="$ORTHO --norm_replay_dir $NORMS" ;;
    normmatch)              TYPE=ddd3r;          EXTRA="$ORTHO --norm_match --gain_log_dir $GAINS" ;;
    prevdir)                TYPE=ddd3r;          EXTRA="$ORTHO --project_prev" ;;
    ttt3r_log)              TYPE=ttt3r;          EXTRA="--gate_log_dir $GATES/ttt3r" ;;
    ttt3r_frozen)           TYPE=ttt3r;          EXTRA="--frozen_gate_dir $GATES/ttt3r" ;;
    ttsa3r_log)             TYPE=ttsa3r;         EXTRA="--gate_log_dir $GATES/ttsa3r" ;;
    ttsa3r_frozen)          TYPE=ttsa3r;         EXTRA="--frozen_gate_dir $GATES/ttsa3r" ;;
    const033)               TYPE=ddd3r_constant; EXTRA="--alpha 0.33" ;;
    const05)                TYPE=ddd3r_constant; EXTRA="--alpha 0.5" ;;
    ttt3r_plain)            TYPE=ttt3r;          EXTRA="" ;;
    brake_t1)               TYPE=ddd3r_brake;    EXTRA="--brake_tau 1.0" ;;
    brake_t2)               TYPE=ddd3r_brake;    EXTRA="--brake_tau 2.0" ;;
    cut3r)                  TYPE=cut3r;          EXTRA="" ;;
    *) echo "Unknown arm: $ARM"; exit 2 ;;
esac

stamp() { date '+%Y-%m-%d %H:%M:%S'; }

if grep -q 'Average ATE' "$OUT/_error_log.txt" 2>/dev/null; then
    echo "$(stamp) SKIP  gpu$GPU $GROUP/$ARM (already complete)"
    exit 0
fi

echo "$(stamp) START gpu$GPU $GROUP/$ARM"
SEQ_ARGS=()
[ ${#SEQS[@]} -gt 0 ] && SEQ_ARGS=(--seq_list "${SEQS[@]}")

CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=${DDD3R_SRC:-src} \
    "${DDD3R_PYTHON:-/home/szy/anaconda3/envs/ttt3r/bin/python}" -m accelerate.commands.launch \
    --num_processes 1 --main_process_port $((29700 + GPU + ${DDD3R_PORT_OFFSET:-0})) \
    eval/relpose/launch.py \
    --weights model/cut3r_512_dpt_4_64.pth --size 512 \
    --output_dir "$OUT" --eval_dataset "$DATASET" \
    --model_update_type "$TYPE" $EXTRA --pose_only "${SEQ_ARGS[@]}" > "$LOG" 2>&1
STATUS=$?

RESULT=$(grep -a 'Average ATE' "$OUT/_error_log.txt" 2>/dev/null | tail -1)
echo "$(stamp) DONE  gpu$GPU $GROUP/$ARM exit=$STATUS ${RESULT:-no average written}"
exit $STATUS
