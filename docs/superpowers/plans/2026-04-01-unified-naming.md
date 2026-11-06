# DDD3R Unified Naming Refactor

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify all code naming to match the paper's DDD3R terminology and unified spectrum narrative.

**Architecture:** Three-layer refactor: (1) model.py update type names + config attrs, (2) CLI arg names in all launch.py files, (3) eval shell scripts. Backward compatibility via alias mapping so old names still work (existing eval results use old directory names). Analysis scripts updated for active ones only.

**Tech Stack:** Python (model code, argparse), Bash (eval scripts)

---

## Naming Mapping

### model_update_type values

| Old Name | New Name | Paper Role |
|----------|----------|------------|
| `cut3r` | `cut3r` | baseline (keep) |
| `ttt3r` | `ttt3r` | existing method (keep) |
| `ttt3r_random` | `ddd3r_constant` | constant dampening (α⊥=α∥=p) |
| `ttt3r_momentum` | `ddd3r_brake` | temporal brake baseline |
| `ttt3r_ortho` | `ddd3r` | directional decomposition |
| `ttt3r_ortho` + `--ortho_adaptive steep` | `ddd3r` + `--gamma >0` | steep adaptive (γ>0 is enough) |
| `ttt3r_constant_brake` | `ddd3r_constant_brake` | combined (叠加实验, keep for completeness) |
| `ttt3r_constant_brake_ortho` | `ddd3r_constant_brake_ortho` | combined (叠加实验, keep for completeness) |
| All abandoned types | unchanged | not renamed, kept as-is |

### CLI argument names

| Old Arg | New Arg | Paper Symbol |
|---------|---------|-------------|
| `--random_gate_p` | `--alpha` | α (for constant dampening, α⊥=α∥) |
| `--ortho_alpha_novel` | `--alpha_perp` | α⊥ |
| `--ortho_alpha_drift` | `--alpha_parallel` | α∥ |
| `--ortho_beta` | `--beta_ema` | β_ema |
| `--ortho_gamma` | `--gamma` | γ |
| `--ortho_adaptive` | removed | steep = γ>0 (no separate flag needed) |
| `--momentum_tau` | `--brake_tau` | τ (brake temperature) |
| `--ortho_warmup_t0` | `--warmup_t0` | T₀ |
| `--ortho_warmup_window` | `--warmup_window` | warmup window |

### Config attributes on model

| Old Attr | New Attr |
|----------|----------|
| `config.random_gate_p` | `config.alpha` |
| `config.ortho_alpha_novel` | `config.alpha_perp` |
| `config.ortho_alpha_drift` | `config.alpha_parallel` |
| `config.ortho_beta` | `config.beta_ema` |
| `config.ortho_gamma` | `config.gamma` |
| `config.ortho_adaptive` | removed (infer from γ>0) |
| `config.momentum_tau` | `config.brake_tau` |
| `config.ortho_warmup_t0` | `config.warmup_t0` |
| `config.ortho_warmup_window` | `config.warmup_window` |

### Output directory naming convention

| Old Pattern | New Pattern |
|-------------|-------------|
| `ttt3r_random` | `ddd3r_constant` |
| `ttt3r_momentum` | `ddd3r_brake` |
| `ttt3r_ortho` | `ddd3r` |
| `ttt3r_ortho_steep_v2_g3` | `ddd3r_g3` |

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/dust3r/model.py` | Modify | Update type aliases + config attr names + getattr fallbacks |
| `eval/relpose/launch.py` | Modify | CLI arg rename + backward compat aliases |
| `eval/video_depth/launch.py` | Modify | CLI arg rename + backward compat aliases |
| `eval/mv_recon/launch.py` | Modify | CLI arg rename + backward compat aliases |
| `eval/run_ddd3r_eval.sh` | Create | New unified eval script with paper naming |
| `CLAUDE.md` | Modify | Update naming table |

---

### Task 1: Add update_type alias mapping in model.py

**Files:**
- Modify: `src/dust3r/model.py`

The key idea: add an alias dict at the top of the update logic, so old names map to new names. Then update the if/elif chains to use new names, with the alias providing backward compat.

- [ ] **Step 1: Add alias mapping and normalize update_type**

In `model.py`, in BOTH `_forward_impl` (around line 898) and `inference_step` (around line 2488), right after `update_type = self.config.model_update_type`, add:

```python
# Backward-compat aliases → canonical DDD3R names
_UPDATE_TYPE_ALIASES = {
    'ttt3r_random': 'ddd3r_constant',
    'ttt3r_momentum': 'ddd3r_brake',
    'ttt3r_ortho': 'ddd3r',
    'ttt3r_constant_brake': 'ddd3r_constant_brake',
    'ttt3r_constant_brake_ortho': 'ddd3r_constant_brake_ortho',
}
update_type = _UPDATE_TYPE_ALIASES.get(update_type, update_type)
```

- [ ] **Step 2: Rename update_type string literals in if/elif chains**

In both `_forward_impl` and `inference_step`, replace:
- `"ttt3r_random"` → `"ddd3r_constant"`
- `"ttt3r_momentum"` → `"ddd3r_brake"`
- `"ttt3r_ortho"` → `"ddd3r"`
- `"ttt3r_constant_brake"` → `"ddd3r_constant_brake"`
- `"ttt3r_constant_brake_ortho"` → `"ddd3r_constant_brake_ortho"`

This includes:
- State initialization block (i==0)
- Main update mask computation (if/elif chain)
- Reset state block (reset_mask.any())

- [ ] **Step 3: Rename config attribute reads with fallbacks**

In `_delta_ortho_update` (~line 1474), change:
```python
alpha_novel = getattr(config, 'alpha_perp', None)
if alpha_novel is None:
    alpha_novel = getattr(config, 'ortho_alpha_novel', 0.5)
alpha_drift = getattr(config, 'alpha_parallel', None)
if alpha_drift is None:
    alpha_drift = getattr(config, 'ortho_alpha_drift', 0.05)
beta = getattr(config, 'beta_ema', None)
if beta is None:
    beta = getattr(config, 'ortho_beta', 0.95)
t0 = getattr(config, 'warmup_t0', None)
if t0 is None:
    t0 = getattr(config, 'ortho_warmup_t0', 0)
warmup_w = getattr(config, 'warmup_window', None)
if warmup_w is None:
    warmup_w = getattr(config, 'ortho_warmup_window', 0)
```

In the `steep` adaptive block, change:
```python
gamma = getattr(config, 'gamma', None)
if gamma is None:
    gamma = getattr(config, 'ortho_gamma', 2.0)
```

Remove `adaptive_mode` check — instead, infer from gamma:
```python
gamma = getattr(config, 'gamma', None)
if gamma is None:
    gamma = getattr(config, 'ortho_gamma', 0.0)
# Also check legacy ortho_adaptive flag
adaptive_mode = getattr(config, 'ortho_adaptive', '')
if gamma > 0 or adaptive_mode == 'steep':
    # steep adaptive path
    ...
elif adaptive_mode in ('linear', 'match', 'threshold'):
    # legacy adaptive modes (keep for backward compat)
    ...
```

In `_momentum_gate` (~line 1410), change:
```python
tau = getattr(config, 'brake_tau', None)
if tau is None:
    tau = getattr(config, 'momentum_tau', 2.0)
```

In the `ddd3r_constant` branch (was `ttt3r_random`), change:
```python
random_p = getattr(config, 'alpha', None)
if random_p is None:
    random_p = getattr(config, 'random_gate_p', 0.5)
```

- [ ] **Step 4: Verify model.py changes compile**

Run:
```bash
cd /home/szy/research/TTT3R && PYTHONPATH=src python -c "from dust3r.model import ARCroco3DStereo; print('OK')"
```
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add src/dust3r/model.py
git commit -m "refactor: rename update types to DDD3R paper naming with backward compat aliases"
```

---

### Task 2: Update CLI arguments in all launch.py files

**Files:**
- Modify: `eval/relpose/launch.py`
- Modify: `eval/video_depth/launch.py`
- Modify: `eval/mv_recon/launch.py`

- [ ] **Step 1: Update eval/relpose/launch.py argparse**

Replace the DDD3R-relevant args with new names + old-name aliases:

```python
# DDD3R unified parameters (paper notation)
parser.add_argument("--alpha", type=float, default=0.5, help="DDD3R constant dampening rate (α⊥=α∥)")
parser.add_argument("--alpha_perp", type=float, default=0.5, help="DDD3R α⊥: novel component coefficient")
parser.add_argument("--alpha_parallel", type=float, default=0.05, help="DDD3R α∥: drift component coefficient")
parser.add_argument("--beta_ema", type=float, default=0.95, help="DDD3R β: EMA decay for drift direction")
parser.add_argument("--gamma", type=float, default=0.0, help="DDD3R γ: steep adaptive exponent (0=fixed ortho, >0=drift-adaptive)")
parser.add_argument("--brake_tau", type=float, default=2.0, help="DDD3R brake temperature")
parser.add_argument("--warmup_t0", type=int, default=0, help="DDD3R: no drift suppression for first T0 frames")
parser.add_argument("--warmup_window", type=int, default=0, help="DDD3R: linear ramp window after T0")

# Backward-compat aliases (hidden, map to new names)
parser.add_argument("--random_gate_p", type=float, default=None, help=argparse.SUPPRESS)
parser.add_argument("--ortho_alpha_novel", type=float, default=None, help=argparse.SUPPRESS)
parser.add_argument("--ortho_alpha_drift", type=float, default=None, help=argparse.SUPPRESS)
parser.add_argument("--ortho_beta", type=float, default=None, help=argparse.SUPPRESS)
parser.add_argument("--ortho_gamma", type=float, default=None, help=argparse.SUPPRESS)
parser.add_argument("--ortho_adaptive", type=str, default="", help=argparse.SUPPRESS)
parser.add_argument("--momentum_tau", type=float, default=None, help=argparse.SUPPRESS)
parser.add_argument("--ortho_warmup_t0", type=int, default=None, help=argparse.SUPPRESS)
parser.add_argument("--ortho_warmup_window", type=int, default=None, help=argparse.SUPPRESS)
```

Remove the old abandoned gate args that are no longer needed for the paper:
- `--gate_base_rate`, `--gate_tau_sharp`, `--novelty_tau`, `--momentum_beta`, `--momentum_lr`
- `--clip_alpha`, `--clip_tau`, `--clip_beta`, `--attn_protect_beta`, `--attn_protect_base`
- `--mem_novelty_base`, `--mem_novelty_tau`, `--mem_novelty_beta`

Keep `--spectral_temperature`, `--geo_gate_tau`, `--geo_gate_freq_cutoff` as-is (abandoned methods but still in model.py).

- [ ] **Step 2: Add alias resolution after parse_args**

After `args = parser.parse_args()`, add a resolution function:

```python
def _resolve_ddd3r_aliases(args):
    """Map old CLI arg names to new ones (backward compat)."""
    if args.random_gate_p is not None and args.alpha == 0.5:
        args.alpha = args.random_gate_p
    if args.ortho_alpha_novel is not None:
        args.alpha_perp = args.ortho_alpha_novel
    if args.ortho_alpha_drift is not None:
        args.alpha_parallel = args.ortho_alpha_drift
    if args.ortho_beta is not None:
        args.beta_ema = args.ortho_beta
    if args.ortho_gamma is not None:
        args.gamma = args.ortho_gamma
    if args.momentum_tau is not None:
        args.brake_tau = args.momentum_tau
    if args.ortho_warmup_t0 is not None:
        args.warmup_t0 = args.ortho_warmup_t0
    if args.ortho_warmup_window is not None:
        args.warmup_window = args.ortho_warmup_window
    # Legacy: ortho_adaptive="steep" → set gamma if not already set
    if args.ortho_adaptive == 'steep' and args.gamma == 0.0:
        args.gamma = args.ortho_gamma if args.ortho_gamma is not None else 2.0
    return args
```

- [ ] **Step 3: Update config assignment block**

Change the model config assignment from old names to new names:

```python
model.config.model_update_type = args.model_update_type
model.config.alpha = args.alpha
model.config.alpha_perp = args.alpha_perp
model.config.alpha_parallel = args.alpha_parallel
model.config.beta_ema = args.beta_ema
model.config.gamma = args.gamma
model.config.brake_tau = args.brake_tau
model.config.warmup_t0 = args.warmup_t0
model.config.warmup_window = args.warmup_window
# Keep for abandoned methods
model.config.spectral_temperature = args.spectral_temperature
model.config.geo_gate_tau = args.geo_gate_tau
model.config.geo_gate_freq_cutoff = args.geo_gate_freq_cutoff
```

- [ ] **Step 4: Apply same changes to eval/video_depth/launch.py**

Same argparse updates, alias resolution, and config assignment changes.

- [ ] **Step 5: Apply same changes to eval/mv_recon/launch.py**

Same argparse updates, alias resolution, and config assignment changes.

- [ ] **Step 6: Verify launch.py files parse correctly**

Run:
```bash
cd /home/szy/research/TTT3R && PYTHONPATH=src python eval/relpose/launch.py --help 2>&1 | head -30
```
Expected: shows new arg names, no errors.

- [ ] **Step 7: Commit**

```bash
git add eval/relpose/launch.py eval/video_depth/launch.py eval/mv_recon/launch.py
git commit -m "refactor: rename CLI args to DDD3R paper notation with backward compat"
```

---

### Task 3: Create unified DDD3R eval script

**Files:**
- Create: `eval/run_ddd3r_eval.sh`

- [ ] **Step 1: Write eval/run_ddd3r_eval.sh**

```bash
#!/bin/bash
# =============================================================================
# DDD3R Unified Evaluation Script (paper naming)
# Usage: bash eval/run_ddd3r_eval.sh [GPU_ID] [DATASET] [METHOD]
#
# Methods: cut3r, ttt3r, ddd3r_constant, ddd3r_brake, ddd3r, ddd3r_g{N}
# Datasets: tum_s1_1000, scannet_s3_1000, sintel, kitti, bonn, 7scenes
# =============================================================================

set -e

GPU=${1:-0}
DATASET=${2:-tum_s1_1000}
METHOD=${3:-ddd3r}

export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=src
PY=/home/szy/anaconda3/envs/ttt3r/bin/python
WEIGHTS="model/cut3r_512_dpt_4_64.pth"
PORT=$((29560 + GPU))

# Parse method → model_update_type + DDD3R params
case "$METHOD" in
    cut3r)
        UPDATE_TYPE="cut3r"
        EXTRA_ARGS=""
        ;;
    ttt3r)
        UPDATE_TYPE="ttt3r"
        EXTRA_ARGS=""
        ;;
    ddd3r_constant)
        UPDATE_TYPE="ddd3r_constant"
        EXTRA_ARGS="--alpha 0.5"
        ;;
    ddd3r_constant_p*)
        # e.g. ddd3r_constant_p033 → alpha=0.33
        P=$(echo "$METHOD" | sed 's/ddd3r_constant_p/0./')
        UPDATE_TYPE="ddd3r_constant"
        EXTRA_ARGS="--alpha $P"
        ;;
    ddd3r_brake)
        UPDATE_TYPE="ddd3r_brake"
        EXTRA_ARGS="--brake_tau 2.0"
        ;;
    ddd3r)
        UPDATE_TYPE="ddd3r"
        EXTRA_ARGS="--alpha_perp 0.5 --alpha_parallel 0.05 --beta_ema 0.95 --gamma 0"
        ;;
    ddd3r_g*)
        # e.g. ddd3r_g2 → gamma=2, ddd3r_g0.5 → gamma=0.5
        GAMMA=$(echo "$METHOD" | sed 's/ddd3r_g//')
        UPDATE_TYPE="ddd3r"
        EXTRA_ARGS="--alpha_perp 0.5 --alpha_parallel 0.05 --beta_ema 0.95 --gamma $GAMMA"
        METHOD="ddd3r_g${GAMMA}"
        ;;
    *)
        echo "Unknown method: $METHOD"
        echo "Available: cut3r, ttt3r, ddd3r_constant, ddd3r_brake, ddd3r, ddd3r_g{N}"
        exit 1
        ;;
esac

# Parse dataset → eval task type
case "$DATASET" in
    tum_*|scannet_*|sintel)
        TASK="relpose"
        LAUNCH="eval/relpose/launch.py"
        ;;
    kitti|bonn|sintel_depth)
        TASK="video_depth"
        LAUNCH="eval/video_depth/launch.py"
        ;;
    7scenes)
        TASK="mv_recon"
        LAUNCH="eval/mv_recon/launch.py"
        ;;
    *)
        echo "Unknown dataset: $DATASET"
        exit 1
        ;;
esac

OUTPUT_DIR="eval_results/${TASK}/${DATASET}/${METHOD}"

echo "=== DDD3R Eval: ${METHOD} on ${DATASET} (GPU ${GPU}) ==="
echo "  update_type: ${UPDATE_TYPE}"
echo "  output_dir:  ${OUTPUT_DIR}"

$PY -m accelerate.commands.launch --num_processes 1 --main_process_port $PORT \
    $LAUNCH \
    --weights $WEIGHTS --size 512 \
    --output_dir $OUTPUT_DIR \
    --eval_dataset $DATASET \
    --model_update_type $UPDATE_TYPE \
    $EXTRA_ARGS

echo "=== Done: ${METHOD} on ${DATASET} ==="
```

- [ ] **Step 2: Make executable**

```bash
chmod +x eval/run_ddd3r_eval.sh
```

- [ ] **Step 3: Commit**

```bash
git add eval/run_ddd3r_eval.sh
git commit -m "feat: add unified DDD3R eval script with paper naming"
```

---

### Task 4: Update CLAUDE.md naming table

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update the Update Types table in CLAUDE.md**

Replace the current table with:

```markdown
## Update Types in model.py

| `model_update_type` | `mask1` (state) | Paper Role | Old Name |
|---------------------|-----------------|------------|----------|
| `cut3r` | 1.0 (baseline) | baseline | — |
| `ttt3r` | sigmoid(cross_attn) | existing method | — |
| `ddd3r_constant` | ttt3r × α | DDD3R with α⊥=α∥=α | `ttt3r_random` |
| `ddd3r_brake` | ttt3r × stability brake | baseline (DDD3R 一阶近似) | `ttt3r_momentum` |
| `ddd3r` | ttt3r_mask + directional decomposition | DDD3R with α⊥>α∥, γ≥0 | `ttt3r_ortho` |
| Others (joint, conf, l2gate, spectral, memgate, etc.) | various | abandoned | — |
```

- [ ] **Step 2: Update eval command examples**

Replace `--model_update_type ttt3r_ortho --ortho_adaptive steep --ortho_gamma 2.0` with:
```
--model_update_type ddd3r --gamma 2.0
```

And the shorthand:
```bash
bash eval/run_ddd3r_eval.sh 0 tum_s1_1000 ddd3r_g2
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md to DDD3R unified naming"
```

---

### Task 5: Verify backward compatibility end-to-end

- [ ] **Step 1: Test old names still work**

```bash
cd /home/szy/research/TTT3R && PYTHONPATH=src python -c "
from dust3r.model import ARCroco3DStereo, ARCroco3DStereoConfig
cfg = ARCroco3DStereoConfig(model_update_type='ttt3r_ortho')
print(f'Old name ttt3r_ortho: OK')
cfg2 = ARCroco3DStereoConfig(model_update_type='ddd3r')
print(f'New name ddd3r: OK')
cfg3 = ARCroco3DStereoConfig(model_update_type='ttt3r_random')
print(f'Old name ttt3r_random: OK')
cfg4 = ARCroco3DStereoConfig(model_update_type='ddd3r_constant')
print(f'New name ddd3r_constant: OK')
"
```

- [ ] **Step 2: Test old CLI args still work**

```bash
cd /home/szy/research/TTT3R && PYTHONPATH=src python eval/relpose/launch.py \
    --weights model/cut3r_512_dpt_4_64.pth --size 512 \
    --model_update_type ttt3r_ortho \
    --ortho_alpha_novel 0.5 --ortho_alpha_drift 0.05 --ortho_beta 0.95 \
    --ortho_adaptive steep --ortho_gamma 2.0 \
    --eval_dataset tum_s1_1000 --help 2>&1 | head -5
```
Expected: no parse error.

- [ ] **Step 3: Test new CLI args work**

```bash
cd /home/szy/research/TTT3R && PYTHONPATH=src python eval/relpose/launch.py \
    --weights model/cut3r_512_dpt_4_64.pth --size 512 \
    --model_update_type ddd3r \
    --alpha_perp 0.5 --alpha_parallel 0.05 --beta_ema 0.95 --gamma 2.0 \
    --eval_dataset tum_s1_1000 --help 2>&1 | head -5
```
Expected: no parse error.

- [ ] **Step 4: Commit verification notes**

No commit needed — this is a verification step.
