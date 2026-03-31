# KITTI Odometry OOD RelPose Evaluation Report

> **Date**: 2026-03-31
> **Dataset**: KITTI Odometry (sequences 00, 02, 05, 07, 08)
> **Setting**: Out-of-Distribution (model trained on ScanNet/TUM indoor, evaluated on KITTI outdoor driving)
> **Metric**: ATE (Absolute Trajectory Error, meters) with Sim(3) Umeyama alignment
> **Hardware**: NVIDIA H200 GPU

## Methods

| Method | Description |
|--------|-------------|
| `cut3r` | Baseline CUT3R — no test-time training |
| `ttt3r` | Sigmoid-gated test-time training |
| `ttt3r_random` | Constant dampening (p=0.33) |
| `ttt3r_momentum` | Stability Brake — adaptive dampening via `alpha = sigmoid(-tau * cos(delta_t, delta_{t-1}))` |
| `ttt3r_ortho` | Delta Orthogonalization — decomposes updates into drift (repeated) vs novel (orthogonal), suppresses drift |

---

## 1. Aggregate Results

### ATE (m) — 200 Frames

| Method | Mean ATE | vs cut3r |
|--------|----------|----------|
| cut3r | 9.454 | — |
| ttt3r | 6.007 | **-36.5%** |
| ttt3r_random | 5.680 | **-39.9%** |
| ttt3r_momentum | 9.543 | +0.9% |
| ttt3r_ortho | 12.421 | +31.4% |

### ATE (m) — 1000 Frames

| Method | Mean ATE | vs cut3r |
|--------|----------|----------|
| cut3r | 106.519 | — |
| ttt3r | 93.409 | -12.3% |
| ttt3r_random | 90.227 | -15.3% |
| ttt3r_momentum | 72.010 | **-32.4%** |
| ttt3r_ortho | 67.876 | **-36.3%** |

---

## 2. Per-Sequence ATE Breakdown

### 200 Frames

| Seq | cut3r | ttt3r | ttt3r_random | ttt3r_momentum | ttt3r_ortho |
|-----|-------|-------|-------------|----------------|-------------|
| 00 | 11.014 | 7.827 | 7.240 | **4.282** | 8.846 |
| 02 | 12.486 | **5.622** | 6.568 | 12.994 | 18.844 |
| 05 | 6.382 | 4.899 | 4.550 | **2.270** | 9.827 |
| 07 | 7.455 | 6.848 | **5.945** | 18.069 | 16.346 |
| 08 | 9.933 | 4.841 | **4.097** | 10.099 | 8.240 |
| **Mean** | **9.454** | **6.007** | **5.680** | **9.543** | **12.421** |

### 1000 Frames

| Seq | cut3r | ttt3r | ttt3r_random | ttt3r_momentum | ttt3r_ortho |
|-----|-------|-------|-------------|----------------|-------------|
| 00 | 112.407 | 121.990 | 117.497 | 100.742 | **65.671** |
| 02 | 187.908 | 118.224 | 128.846 | 87.094 | **63.728** |
| 05 | 73.879 | 72.749 | 56.829 | **48.212** | 55.953 |
| 07 | 60.708 | 60.419 | 63.614 | **54.389** | 62.607 |
| 08 | 97.691 | 93.662 | 84.346 | **69.612** | 91.423 |
| **Mean** | **106.519** | **93.409** | **90.227** | **72.010** | **67.876** |

---

## 3. RPE (Relative Pose Error) Summary

### RPE Rotation (deg/frame) — Mean across sequences

| Method | 200f | 1000f |
|--------|------|-------|
| cut3r | 0.779 | 1.042 |
| ttt3r | 0.647 | 1.291 |
| ttt3r_random | 0.633 | 1.545 |
| ttt3r_momentum | 0.828 | 2.559 |
| ttt3r_ortho | 0.691 | 1.628 |

### RPE Translation (m/frame) — Mean across sequences

| Method | 200f | 1000f |
|--------|------|-------|
| cut3r | 0.668 | 1.441 |
| ttt3r | 0.603 | 1.796 |
| ttt3r_random | 0.688 | 2.305 |
| ttt3r_momentum | 1.587 | 4.940 |
| ttt3r_ortho | 1.440 | 3.720 |

---

## 4. Analysis

### Key Finding: Sequence Length Determines Optimal Dampening Strategy

The results reveal a striking **crossover effect** between short (200f) and long (1000f) sequences:

- **Short sequences (200f)**: Simple methods win. `ttt3r_random` (constant p=0.33) achieves the best mean ATE at 5.680m (-39.9%), followed closely by `ttt3r` (sigmoid gating) at 6.007m (-36.5%). Both adaptive methods (`momentum` and `ortho`) **fail**, performing at or worse than the baseline.

- **Long sequences (1000f)**: Adaptive methods dominate. `ttt3r_ortho` achieves the best mean ATE at 67.876m (-36.3%), followed by `ttt3r_momentum` at 72.010m (-32.4%). Simple dampening methods show smaller gains.

### Why Adaptive Methods Fail on Short Sequences

Adaptive methods (`momentum`, `ortho`) need a **burn-in period** to build up meaningful statistics:
- `ttt3r_momentum` requires past deltas to compute alignment cosine — early frames have unreliable signals
- `ttt3r_ortho` accumulates a drift subspace via exponential moving average (beta=0.95) — insufficient history in 200 frames leads to incorrect drift/novel decomposition

This manifests as extreme variance: `ttt3r_momentum` scores 2.270 on seq 05 (best overall) but 18.069 on seq 07 (worst overall). The adaptive signal occasionally helps but is too noisy to be reliable.

### Why Adaptive Methods Excel on Long Sequences

Over 1000 frames, the **over-update problem** becomes severe — repeated gradient steps in similar directions cause state drift. This is exactly what adaptive dampening targets:

- `ttt3r_ortho` suppresses the drift component of updates (alpha_drift=0.05 vs alpha_novel=0.5), preserving only novel information. On the hardest sequence (02, highway driving with 187.9m baseline ATE), ortho reduces error by **66.1%** to 63.7m.
- `ttt3r_momentum` detects aligned consecutive updates (likely drift) and dampens them. It achieves consistent improvements across all 5 sequences.

### RPE vs ATE Disconnect

An interesting pattern: methods with the best ATE (global trajectory) often have higher RPE (frame-to-frame error). For 1000f, `ttt3r_momentum` has the highest RPE_trans (4.940 m/frame) but second-best ATE. This suggests adaptive dampening trades local accuracy for better global consistency — it prevents cumulative drift even at the cost of noisier per-frame estimates.

### Sequence Difficulty Ranking

By baseline (cut3r) ATE:
- **Hardest**: seq 02 (highway, long straight) — 12.5m (200f), 187.9m (1000f)
- **Medium**: seq 00, 08 (mixed urban/suburban)
- **Easiest**: seq 05, 07 (residential) — 6.4m (200f), 60.7m (1000f)

Adaptive methods show the largest gains on the hardest sequences, where cumulative drift is most pronounced.

---

## 5. Conclusions

1. **Test-time training consistently helps** on OOD data — all TTT variants improve over `cut3r` on 1000-frame sequences
2. **Sequence length is the critical variable** for choosing dampening: short sequences favor simple constant dampening, long sequences strongly favor adaptive methods
3. **Delta Orthogonalization is the best long-sequence method**, achieving -36.3% ATE reduction on 1000f with especially strong gains on high-drift sequences
4. **Stability Brake is the most consistent adaptive method**, improving on all 5 sequences at 1000f (no regression on any sequence)
5. The **burn-in problem** of adaptive methods is a clear area for improvement — combining a constant dampening warmup with adaptive methods could yield the best of both worlds

---

## Appendix: Experiment Configuration

```
Dataset: KITTI Odometry (outdoor driving, stereo)
Sequences: 00, 02, 05, 07, 08 (with ground truth)
Frame lengths: 200, 1000
Stride: 1 (every frame)
Model: cut3r_512_dpt_4_64.pth
Alignment: Sim(3) Umeyama
Evaluation: evo library (ATE + RPE)
GPU: NVIDIA H200
```
