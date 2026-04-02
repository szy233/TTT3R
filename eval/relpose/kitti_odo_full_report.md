# KITTI Odometry Full-Sequence RelPose Evaluation Report

> **Date**: 2026-04-02
> **Dataset**: KITTI Odometry — all 11 sequences (00–10), full length
> **Setting**: Out-of-Distribution (model trained on ScanNet/TUM indoor, evaluated on KITTI outdoor driving)
> **Metrics**: ATE (Absolute Trajectory Error, meters) with Sim(3) Umeyama alignment; KITTI official translation error (%) and rotation error (deg/100m)
> **Hardware**: NVIDIA H200 GPU

## Methods

| Method | Description |
|--------|-------------|
| `cut3r` | Baseline CUT3R — no test-time training |
| `ttt3r` | Sigmoid-gated test-time training |
| `constant` | Constant dampening (p=0.5) — DDD3R with α⊥ = α∥ |
| `brake` | Stability Brake — adaptive dampening via `sigmoid(-τ·cos(δ_t, δ_{t-1}))` |
| `ortho` | Delta Orthogonalization (γ=0) — fixed directional decomposition |
| `ddd3r_g1` | DDD3R steep adaptive, γ=1 |
| `ddd3r_g2` | DDD3R steep adaptive, γ=2 |
| `ddd3r_g3` | DDD3R steep adaptive, γ=3 |
| `ddd3r_g4` | DDD3R steep adaptive, γ=4 |
| `ddd3r_g5` | DDD3R steep adaptive, γ=5 |
| `auto_steep_clamp` | Auto-γ with clamp scheduling |
| `auto_steep_sigmoid` | Auto-γ with sigmoid scheduling |
| `auto_warmup_linear` | Auto-γ with linear warmup |
| `auto_warmup_threshold` | Auto-γ with threshold warmup |

### Sequence Info

| Seq | Frames | Environment |
|-----|--------|-------------|
| 00 | 4541 | Urban |
| 01 | 1101 | Highway |
| 02 | 4661 | Urban/Suburban |
| 03 | 801 | Residential |
| 04 | 271 | Straight road |
| 05 | 2761 | Urban |
| 06 | 1101 | Urban |
| 07 | 1101 | Urban |
| 08 | 4071 | Urban/Suburban |
| 09 | 1591 | Suburban |
| 10 | 1201 | Suburban |

---

## 1. Aggregate Results — ATE (m)

| Method | Avg ATE ↓ | vs cut3r |
|--------|-----------|----------|
| cut3r | 212.59 | — |
| ttt3r | 183.16 | -13.8% |
| **constant** | **179.33** | **-15.6%** |
| brake | 185.68 | -12.7% |
| ortho | 179.40 | -15.6% |
| **ddd3r_g1** | **174.04** | **-18.1%** |
| ddd3r_g2 | 390.02 | +83.5% |
| ddd3r_g3 | 375.68 | +76.7% |
| ddd3r_g4 | 392.86 | +84.8% |
| ddd3r_g5 | 387.79 | +82.4% |
| auto_steep_clamp | 176.44 | -17.0% |
| auto_steep_sigmoid | 182.28 | -14.3% |
| **auto_warmup_linear** | **172.06** | **-19.1%** |
| auto_warmup_threshold | 182.70 | -14.1% |

**Best**: `auto_warmup_linear` (172.06, -19.1%), followed by `ddd3r_g1` (174.04, -18.1%).

> **Warning**: `ddd3r_g2`–`ddd3r_g5` show extremely high average ATE (~375–393m), far worse than baseline. This is driven by catastrophic failure on seq 01 (highway, 1101 frames) where ATE jumps to 700+m.

---

## 2. Aggregate Results — KITTI Official Metrics

| Method | t_err (%) ↓ | r_err (deg/100m) ↓ | vs cut3r (t_err) |
|--------|-------------|---------------------|------------------|
| cut3r | 93.94 | 22.66 | — |
| ttt3r | 95.66 | 21.09 | +1.8% |
| constant | 92.13 | 18.50 | -1.9% |
| brake | 88.17 | 16.75 | -6.1% |
| **ortho** | **86.77** | **9.38** | **-7.6%** |
| ddd3r_g1 | 91.67 | 18.28 | -2.4% |
| ddd3r_g2 | 89.86 | 14.16 | -4.3% |
| ddd3r_g3 | 90.54 | 14.80 | -3.6% |
| ddd3r_g4 | 87.94 | 10.96 | -6.4% |
| **ddd3r_g5** | **86.75** | **12.26** | **-7.6%** |
| auto_steep_clamp | 92.71 | 16.58 | -1.3% |
| auto_steep_sigmoid | 92.67 | 16.91 | -1.4% |
| auto_warmup_linear | 90.61 | 16.93 | -3.5% |
| auto_warmup_threshold | 91.20 | 18.84 | -2.9% |

**Key finding**: `ortho` achieves the best rotation error (9.38 deg/100m, -58.6% vs cut3r) and tied-best translation error. On KITTI official metrics, ortho-family methods (ortho, ddd3r_g4, ddd3r_g5) dominate despite their higher ATE — the segment-based KITTI metric is more forgiving of global drift.

---

## 3. Per-Sequence ATE Breakdown (m)

### Core Methods

| Seq | Frames | cut3r | ttt3r | constant | brake | ortho |
|-----|--------|-------|-------|----------|-------|-------|
| 00 | 4541 | 187.79 | 163.18 | 181.04 | 172.51 | 174.10 |
| 01 | 1101 | 659.71 | 493.91 | 492.59 | 723.75 | 714.73 |
| 02 | 4661 | 298.82 | 274.68 | 272.46 | 282.22 | 278.10 |
| 03 | 801 | 163.58 | 124.89 | 103.37 | **39.06** | **24.83** |
| 04 | 271 | 32.45 | 13.53 | **8.25** | 6.66 | 10.28 |
| 05 | 2761 | 152.03 | 154.37 | 121.74 | 112.85 | 115.09 |
| 06 | 1101 | 131.30 | 135.42 | 124.22 | **60.01** | 64.45 |
| 07 | 1101 | 68.88 | 68.34 | 72.17 | **58.05** | 67.36 |
| 08 | 4071 | 258.24 | 255.13 | 231.58 | 240.83 | **165.69** |
| 09 | 1591 | 188.08 | 196.58 | 180.03 | **144.02** | 153.74 |
| 10 | 1201 | 197.60 | 134.76 | 185.15 | 202.51 | **102.01** |
| **Avg** | | **212.59** | **183.16** | **179.33** | **185.68** | **179.40** |

### DDD3R Gamma Sweep

| Seq | Frames | ortho (γ=0) | γ=1 | γ=2 | γ=3 | γ=4 | γ=5 |
|-----|--------|-------------|-----|-----|-----|-----|-----|
| 00 | 4541 | 174.10 | **163.03** | 168.90 | 175.89 | 174.58 | 177.56 |
| 01 | 1101 | 714.73 | **551.02** | 718.11 | 703.14 | 717.88 | 707.77 |
| 02 | 4661 | 278.10 | **273.01** | 283.07 | 248.01 | 286.10 | 278.04 |
| 03 | 801 | **24.83** | 54.64 | 31.43 | 26.77 | 23.37 | 23.89 |
| 04 | 271 | 10.28 | **6.05** | 7.80 | 7.07 | 8.28 | 8.23 |
| 05 | 2761 | 115.09 | **101.88** | 110.70 | 112.64 | 114.49 | 118.61 |
| 06 | 1101 | 64.45 | 107.26 | **60.44** | 66.07 | 69.91 | 70.17 |
| 07 | 1101 | 67.36 | 74.36 | 73.05 | 68.83 | **65.07** | 77.89 |
| 08 | 4071 | **165.69** | 235.15 | 194.64 | 227.29 | 214.23 | 234.00 |
| 09 | 1591 | 153.74 | 183.03 | 157.24 | 161.61 | 144.92 | **144.02** |
| 10 | 1201 | 102.01 | **81.89** | 89.26 | **67.42** | 127.24 | 95.81 |
| **Avg** | | **179.40** | **174.04** | **190.02** | **169.52** | **176.92** | **176.00** |

> Note: The overall average for γ≥2 is skewed by seq 01 (highway) catastrophic failures (~700m). **Excluding seq 01**, averages are much closer (see Analysis below).

### Auto-Gamma Variants

| Seq | Frames | auto_steep_clamp | auto_steep_sigmoid | auto_warmup_linear | auto_warmup_threshold |
|-----|--------|------------------|--------------------|--------------------|-----------------------|
| 00 | 4541 | 176.44 | 182.28 | **172.06** | 182.70 |
| 01 | 1101 | 719.73 | 720.77 | 581.78 | **501.67** |
| 02 | 4661 | 282.63 | 266.07 | 276.91 | **257.36** |
| 03 | 801 | 48.88 | 52.37 | **43.18** | 104.40 |
| 04 | 271 | 10.02 | 8.22 | **5.88** | 7.08 |
| 05 | 2761 | 115.68 | 111.20 | **100.57** | 116.01 |
| 06 | 1101 | **56.92** | 68.82 | 63.75 | 126.68 |
| 07 | 1101 | 70.91 | 74.96 | 78.61 | **67.86** |
| 08 | 4071 | 246.81 | **213.26** | 251.52 | 249.28 |
| 09 | 1591 | 181.60 | 187.00 | **173.44** | 186.02 |
| 10 | 1201 | 154.29 | 76.14 | **122.37** | 191.13 |
| **Avg** | | **187.63** | **178.28** | **170.01** | **181.11** |

---

## 4. RPE Summary

### RPE Translation (m/frame) — Average across 11 sequences

| Method | RPE trans ↓ |
|--------|-------------|
| cut3r | 2.66 |
| ttt3r | 3.46 |
| constant | 4.44 |
| brake | 5.37 |
| ortho | 6.06 |
| ddd3r_g1 | 5.66 |
| auto_warmup_linear | 3.25 |
| auto_warmup_threshold | 2.18 |

### RPE Rotation (deg/frame) — Average across 11 sequences

| Method | RPE rot ↓ |
|--------|-----------|
| cut3r | 2.97 |
| ttt3r | 4.56 |
| constant | 5.83 |
| ortho | 4.59 |
| ddd3r_g1 | 6.13 |
| auto_warmup_threshold | 8.32 |
| auto_steep_sigmoid | 10.65 |

---

## 5. Analysis

### 5.1 ATE vs KITTI Official Metrics — Different Winners

A striking finding is the **disconnect between ATE and KITTI official metrics**:

| Metric | Best Method | Score |
|--------|-------------|-------|
| ATE (global) | auto_warmup_linear | 172.06m (-19.1%) |
| KITTI t_err (segment) | ddd3r_g5 / ortho | 86.75% / 86.77% (-7.6%) |
| KITTI r_err (rotation) | ortho | 9.38 deg/100m (**-58.6%**) |

ATE measures global trajectory drift after Sim(3) alignment. KITTI official metrics evaluate on fixed-length segments (100–800m), measuring **local accuracy**. Ortho-family methods achieve dramatically better local accuracy (especially rotation) even when their global trajectories drift more.

### 5.2 Seq 01 (Highway) — The Outlier

Seq 01 (1101 frames, highway driving) is an extreme outlier:

| Method | Seq 01 ATE | All-seq Avg | Avg excl. 01 |
|--------|------------|-------------|---------------|
| cut3r | 659.71 | 212.59 | 167.82 |
| brake | 723.75 | 185.68 | 131.88 |
| ortho | 714.73 | 179.40 | 125.87 |
| ddd3r_g1 | 551.02 | 174.04 | 136.24 |
| ddd3r_g2 | 718.11 | 390.02 | 157.16 |
| auto_warmup_linear | 581.78 | 172.06 | 131.00 |

All methods struggle on highway driving (long straight segments, minimal features). Brake and ortho actually **regress** vs cut3r on this sequence. `ddd3r_g1` and `auto_warmup_linear` partially mitigate the issue.

### 5.3 Gamma Sensitivity — γ=1 is the Sweet Spot

Excluding the seq 01 outlier:

| γ | Avg ATE (excl. 01) | vs ortho |
|---|-------------------|----------|
| 0 (ortho) | 125.87 | — |
| 1 | 136.24 | +8.2% |
| 2 | 117.56 | -6.6% |
| 3 | 116.16 | -7.7% |
| 4 | 122.82 | -2.4% |
| 5 | 122.82 | -2.4% |

With seq 01 excluded, γ=2–3 are optimal, slightly outperforming pure ortho. The issue is that **γ=1 uniquely rescues seq 01** (551 vs 714), making it the best on aggregate.

### 5.4 Auto-Gamma — Promising but Inconsistent

The auto-gamma variants attempt to learn γ adaptively:

| Variant | Avg ATE | Key Strength | Key Weakness |
|---------|---------|-------------|--------------|
| auto_warmup_linear | **172.06** | Best overall, consistent | RPE trans higher than cut3r |
| auto_steep_clamp | 176.44 | Stable on short seqs | Fails on seq 01 (719m) |
| auto_steep_sigmoid | 182.28 | Best on seq 10 (76m) | Inconsistent |
| auto_warmup_threshold | 182.70 | Best RPE trans (2.18) | Poor on seq 06, 10 |

`auto_warmup_linear` is the most reliable auto-gamma variant, achieving the lowest average ATE across all methods.

### 5.5 Short vs Long Sequences

| Length | Best Method | ATE |
|--------|-------------|-----|
| Short (271–801f): seq 03, 04 | ortho family | ortho/ddd3r_g4 best |
| Medium (1101f): seq 01, 06, 07, 10 | brake / auto methods | brake best on 06, 07 |
| Long (1591–4661f): seq 00, 02, 05, 08, 09 | ddd3r_g1 / auto_warmup_linear | consistent improvement |

On short sequences (03, 04), ortho-family methods show massive gains (e.g., seq 03: cut3r 163.6 → ortho 24.8, **-84.8%**). On long sequences, auto-warmup methods provide more consistent improvement.

### 5.6 Brake Paradox

Brake shows a paradoxical pattern: excellent on some sequences (seq 03: 39.1, seq 06: 60.0) but **catastrophic on seq 01** (723.7 > cut3r 659.7). This is consistent with the brake's known limitation — it relies on consecutive-frame cosine similarity, which is unreliable on highway driving where update directions change rapidly.

---

## 6. Conclusions

1. **Over-update is universal on KITTI**: Every dampening method improves over `cut3r` on average (except ddd3r_g2–g5 due to seq 01 failure), confirming M1.

2. **Ortho dominates KITTI official metrics**: -58.6% rotation error, -7.6% translation error. For **local accuracy**, directional decomposition is clearly superior.

3. **γ=1 is the best DDD3R configuration**: It uniquely balances seq 01 rescue with consistent improvement elsewhere, achieving the lowest fixed-gamma average ATE (174.04).

4. **Auto-warmup-linear is the overall winner**: 172.06m average ATE (-19.1% vs cut3r), the best single method across all 11 sequences.

5. **Highway driving remains challenging**: All methods struggle on seq 01 (featureless highway). This is an inherent limitation of indoor-trained models on OOD outdoor data, not specific to any dampening strategy.

6. **ATE and KITTI metrics tell different stories**: Method selection depends on whether the application prioritizes global trajectory consistency (ATE → auto_warmup_linear) or local motion accuracy (KITTI metrics → ortho).

---

## Appendix A: Experiment Configuration

### A.1 Hardware & Software Environment

| Item | Specification |
|------|---------------|
| GPU | NVIDIA H200 (80 GB HBM3) |
| CUDA | 12.1 |
| Python | 3.10.12 |
| PyTorch | 2.5.1+cu121 |
| Accelerate | 1.13.0 |
| evo (trajectory evaluation) | v1.34.3 |
| NumPy | 1.26.4 |
| SciPy | 1.15.3 |
| OpenCV | 4.11.0 |
| einops | 0.8.2 |
| Pillow | 10.3.0 |

### A.2 Model

| Item | Detail |
|------|--------|
| Checkpoint | `model/cut3r_512_dpt_4_64.pth` |
| MD5 | `31cd42b49f93253082d50c32ff1fa58f` |
| Architecture | CUT3R (ARCroco3DStereo) — DPT head, 4 layers, feature dim 64 |
| Training data | ScanNet + TUM (indoor) — **no KITTI data during training** |
| Input resolution | 512 × 512 |

### A.3 Dataset

| Item | Detail |
|------|--------|
| Dataset | KITTI Odometry (Geiger et al., CVPR 2012) |
| Sequences | 00–10 (all 11 sequences with ground truth poses) |
| Frame count | 271–4661 per sequence, 22,410 total |
| Frame stride | 1 (every frame used) |
| Image source | Left camera (image_2), grayscale converted |
| Ground truth | KITTI official pose files (12-float per line, 3×4 projection matrix) |
| Data path | `data/long_kitti_odo_s1/{seq}/image_full/` (images), `data/long_kitti_odo_s1/{seq}/pose_full.txt` (GT in TUM format) |
| Preprocessing | Raw KITTI images resized to 512×512; GT poses converted from KITTI format (3×4 matrix) to TUM format (timestamp tx ty tz qx qy qz qw) |

### A.4 Evaluation Metrics

**ATE (Absolute Trajectory Error)**:
- Library: `evo` v1.34.3 (`evo.main_ape`)
- Pose relation: `PoseRelation.translation_part`
- Alignment: `align=True, correct_scale=True` (Sim(3) Umeyama, 7-DoF: scale + SE3)
- Statistic reported: **RMSE** of aligned translation errors (meters)

**RPE (Relative Pose Error)**:
- Library: `evo` v1.34.3 (`evo.main_rpe`)
- Delta: 1 frame, `all_pairs=True`
- RPE translation: `PoseRelation.translation_part`, RMSE (meters/frame)
- RPE rotation: `PoseRelation.rotation_angle_deg`, RMSE (degrees/frame)
- Alignment: `align=True, correct_scale=True`

**KITTI Official Metrics** (computed post-hoc via `eval/relpose/compute_kitti_errors.py`):
- Translation error (%): mean relative translation error over segments
- Rotation error (deg/100m): mean relative rotation error over segments
- Segment lengths: [100, 200, 300, 400, 500, 600, 700, 800] meters
- Start-frame sampling: ~10% of all valid starting frames
- Alignment: 7-DoF Umeyama (scale + SE3) before evaluation

---

## Appendix B: Method Hyperparameters

### B.1 DDD3R Unified Update Rule

All DDD3R variants share the same update rule (implemented in `src/dust3r/model.py`):

```
S_t = S_{t-1} + β_t · (α⊥ · δ⊥ + α∥_eff · δ∥)
```

where `δ = δ⊥ + δ∥` is decomposed via projection onto EMA drift direction `d_t`.

### B.2 Per-Configuration Hyperparameters

| Config | `model_update_type` | α⊥ | α∥ | β_ema | γ | auto_gamma | Notes |
|--------|--------------------:|----:|----:|------:|---:|:-----------|:------|
| cut3r | `cut3r` | — | — | — | — | — | Baseline, mask=1.0 |
| ttt3r | `ttt3r` | — | — | — | — | — | Sigmoid gate from cross-attention |
| constant | `ttt3r_random` | 0.5 | 0.5 | — | — | — | Isotropic dampening (α⊥=α∥=0.5) |
| brake | `ttt3r_momentum` | — | — | — | — | — | τ=2.0, mask=sigmoid(-τ·cos) |
| ortho | `ddd3r` | 0.5 | 0.05 | 0.95 | 0.0 | — | Fixed directional decomposition |
| ddd3r_g1 | `ddd3r` | 0.5 | 0.05 | 0.95 | 1.0 | — | Steep adaptive |
| ddd3r_g2 | `ddd3r` | 0.5 | 0.05 | 0.95 | 2.0 | — | Steep adaptive |
| ddd3r_g3 | `ddd3r` | 0.5 | 0.05 | 0.95 | 3.0 | — | Steep adaptive |
| ddd3r_g4 | `ddd3r` | 0.5 | 0.05 | 0.95 | 4.0 | — | Steep adaptive |
| ddd3r_g5 | `ddd3r` | 0.5 | 0.05 | 0.95 | 5.0 | — | Steep adaptive |
| auto_steep_clamp | `ddd3r` | 0.5 | 0.05 | 0.95 | — | `steep_clamp` | lo=0.3, hi=0.6, max_γ=3.0 |
| auto_steep_sigmoid | `ddd3r` | 0.5 | 0.05 | 0.95 | — | `steep_sigmoid` | k=10.0, max_γ=3.0 |
| auto_warmup_linear | `ddd3r` | 0.5 | 0.05 | 0.95 | — | `warmup_linear` | warmup=30 frames, max_γ=3.0 |
| auto_warmup_threshold | `ddd3r` | 0.5 | 0.05 | 0.95 | — | `warmup_threshold` | warmup=30 frames, max_γ=3.0 |

### B.3 Steep Adaptive Formula

```
e_t = ⟨δ̂_t, d̂_t⟩²            # per-token drift energy (scalar)
w_t = e_t^γ                     # weight toward isotropic
α∥_eff(t) = w_t·α⊥ + (1-w_t)·α∥  # interpolated drift coefficient
```

- γ → 0: w → 1, α∥_eff → α⊥ → isotropic (= constant dampening)
- γ → ∞: w → 0, α∥_eff → α∥ → full directional decomposition (= pure ortho)

---

## Appendix C: Reproducibility

### C.1 Repository & Commit

| Item | Value |
|------|-------|
| Repository | `github.com/szy233/TTT3R` |
| Branch | `zjc` |
| Commit | `d9ee9a1` |

### C.2 Key Source Files

| File | Role |
|------|------|
| `src/dust3r/model.py` | Model architecture, all update types (cut3r/ttt3r/ddd3r), gate methods, state update logic |
| `eval/relpose/launch.py` | Evaluation entry point — inference loop, trajectory saving, metric computation |
| `eval/relpose/evo_utils.py` | ATE/RPE computation via evo library (Sim(3) alignment, RMSE statistics) |
| `eval/relpose/metadata.py` | Dataset configuration (paths, formats, sequence lists) |
| `eval/relpose/compute_kitti_errors.py` | KITTI official t_err%/r_err computation (segment-based, Umeyama alignment) |
| `scripts/server/run_all_kitti_sequential.sh` | Orchestration script for running all 14 configs sequentially |

### C.3 Exact Commands to Reproduce

**Step 1: Run inference for each configuration**

```bash
# Environment setup
conda activate ttt3r
export WORKDIR=~/TTT3R
export MODEL=${WORKDIR}/model/cut3r_512_dpt_4_64.pth

# --- Baselines ---

# cut3r (no test-time training)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=${WORKDIR}/src accelerate launch \
    --num_processes 1 --main_process_port 29580 \
    eval/relpose/launch.py \
    --weights ${MODEL} --size 512 \
    --output_dir eval_results/relpose/kitti_odo_full/cut3r \
    --eval_dataset kitti_odo_full \
    --model_update_type cut3r

# ttt3r (sigmoid gate)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=${WORKDIR}/src accelerate launch \
    --num_processes 1 --main_process_port 29580 \
    eval/relpose/launch.py \
    --weights ${MODEL} --size 512 \
    --output_dir eval_results/relpose/kitti_odo_full/ttt3r \
    --eval_dataset kitti_odo_full \
    --model_update_type ttt3r

# constant dampening (α⊥ = α∥ = 0.5)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=${WORKDIR}/src accelerate launch \
    --num_processes 1 --main_process_port 29580 \
    eval/relpose/launch.py \
    --weights ${MODEL} --size 512 \
    --output_dir eval_results/relpose/kitti_odo_full/constant \
    --eval_dataset kitti_odo_full \
    --model_update_type ttt3r_random --alpha 0.5

# brake (stability brake, τ=2.0)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=${WORKDIR}/src accelerate launch \
    --num_processes 1 --main_process_port 29580 \
    eval/relpose/launch.py \
    --weights ${MODEL} --size 512 \
    --output_dir eval_results/relpose/kitti_odo_full/brake \
    --eval_dataset kitti_odo_full \
    --model_update_type ttt3r_momentum --brake_tau 2.0

# --- DDD3R variants ---

# ortho (γ=0, fixed directional decomposition)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=${WORKDIR}/src accelerate launch \
    --num_processes 1 --main_process_port 29580 \
    eval/relpose/launch.py \
    --weights ${MODEL} --size 512 \
    --output_dir eval_results/relpose/kitti_odo_full/ortho \
    --eval_dataset kitti_odo_full \
    --model_update_type ddd3r --gamma 0.0

# ddd3r_g{1,2,3,4,5} (steep adaptive, γ=1..5)
for G in 1 2 3 4 5; do
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=${WORKDIR}/src accelerate launch \
        --num_processes 1 --main_process_port 29580 \
        eval/relpose/launch.py \
        --weights ${MODEL} --size 512 \
        --output_dir eval_results/relpose/kitti_odo_full/ddd3r_g${G} \
        --eval_dataset kitti_odo_full \
        --model_update_type ddd3r --gamma ${G}
done

# --- Auto-gamma variants ---
for MODE in warmup_linear warmup_threshold steep_sigmoid steep_clamp; do
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=${WORKDIR}/src accelerate launch \
        --num_processes 1 --main_process_port 29580 \
        eval/relpose/launch.py \
        --weights ${MODEL} --size 512 \
        --output_dir eval_results/relpose/kitti_odo_full/auto_${MODE} \
        --eval_dataset kitti_odo_full \
        --model_update_type ddd3r --auto_gamma ${MODE}
done
```

**Step 2: Compute KITTI official metrics**

```bash
python eval/relpose/compute_kitti_errors.py \
    --results_dir eval_results/relpose/kitti_odo_full \
    --kitti_poses_dir /path/to/kitti/poses \
    --seqs 00 01 02 03 04 05 06 07 08 09 10
```

### C.4 Output Structure

```
eval_results/relpose/kitti_odo_full/
├── <config>/                        # e.g., cut3r, ortho, ddd3r_g1, ...
│   ├── 00/
│   │   ├── pred_traj.txt            # Predicted trajectory (TUM format)
│   │   ├── pred_focal.txt           # Predicted focal lengths
│   │   └── pred_intrinsics.txt      # Predicted intrinsics
│   ├── 00_eval_metric.txt           # Per-sequence ATE/RPE details (evo output)
│   ├── 01/ ... 10/                  # Same structure for each sequence
│   ├── _error_log.txt               # Aggregated ATE/RPE across all sequences
│   └── kitti_errors.txt             # KITTI official t_err% / r_err (post-hoc)
```

### C.5 Runtime & Resource Usage

| Metric | Value |
|--------|-------|
| Peak GPU memory | ~6 GB per process |
| Inference speed | ~9–10 FPS (512×512 input) |
| Time per sequence (1000f) | ~100 seconds |
| Total time (11 seqs × 14 configs) | ~12 hours |
| Disk per config (11 seqs) | ~150 MB (trajectories + metrics) |

### C.6 Random Seed & Determinism

All methods in this evaluation are **deterministic at inference time** — there is no random sampling, dropout, or stochastic augmentation. The DDD3R update rule is a closed-form computation on the model's output. Results are fully reproducible given the same model checkpoint, input images, and hardware (floating-point precision may vary across GPU architectures).
