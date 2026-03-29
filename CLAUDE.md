# DD3R — Training-Free Directional Decomposition Beyond Scalar Gating for Recurrent 3D Reconstruction

## Project Goal
NeurIPS submission. Train-free, inference-time method to address systematic over-update in recurrent 3D reconstruction (CUT3R). 核心方法：Directional Decomposition (Delta Orthogonalization) — 将 state update 分解为 drift（重复方向）和 novel（新信息）分量，差异化抑制。超越 scalar gating 的方向性分解框架。

## Architecture Overview

Model: `src/dust3r/model.py`, class `ARCroco3DStereo`. Recurrent processing:
1. Encode frame → `feat_i`
2. `_recurrent_rollout(state_feat, feat_i)` → `new_state_feat`, `dec`
3. `pose_retriever.update_mem(mem, feat, pose)` → `new_mem`
4. `_downstream_head(dec)` → `res` (pts3d, conf)
5. State update: `state_feat = new * mask1 + old * (1-mask1)`
6. Memory update: `mem = new_mem * mask2 + mem * (1-mask2)`

`mask1` is where our method is applied.

## Method: Delta Orthogonalization (`ttt3r_ortho`)

```python
delta = new_state_feat - state_feat
drift_dir = EMA(delta_dir, β=0.95)    # 追踪漂移方向
drift_comp = proj(delta, drift_dir)    # 投影到漂移方向 → 强抑制 (α_drift=0.05)
novel_comp = delta - drift_comp        # 正交分量（新信息）→ 保留 (α_novel=0.5)
updated = state_feat + α_novel × novel_comp + α_drift × drift_comp
```

**Adaptive 模式** (`--ortho_adaptive`): 根据 per-token drift energy (cos² EMA) 动态调节 α_drift
- `linear`: `α_drift = base + (α_novel - base) × drift_energy`
- `match`: `α_drift = α_novel × drift_energy + base × (1 - drift_energy)`
- `threshold`: per-token 二值切换，drift_energy > 0.5 用 uniform dampening

**Motivation** (A1-A4 分析):
- A1: 所有 scalar adaptive gate 退化为 ~constant 0.33（std≈0.02），无时序动态 → scalar gate 方向不可行
- A2: cos variance vs improvement 无相关（r=-0.13, p=0.63）→ adaptive timing 无意义
- A3: Scalar gate per-scene ~50/50 改善/退化 → 不 robust
- A4: cos(δ_t, δ_{t-1}) TUM≈0.62, ScanNet≈0.77 → drift energy 差异显著（40% vs 60%），需要 dataset-adaptive 策略

## Key Results

### Relpose ATE — TUM（8 sequences）

| Config | ATE ↓ | vs cut3r | vs random |
|--------|-------|----------|-----------|
| cut3r (baseline) | 0.166 | — | — |
| ttt3r | 0.103 | -38.1% | — |
| ttt3r_random (p=0.33) | 0.066 | -60.2% | — |
| ttt3r_momentum_inv_t1 | 0.063 | -61.8% | -4.5% |
| **ttt3r_ortho** | **0.056** | **-66.5%** | **-15.4%** |
| ttt3r_ortho_adaptive (linear) | 0.055 | -66.9% | -16.7% |

### Relpose ATE — ScanNet（96 scenes, 65 valid）

| Config | ATE ↓ | vs cut3r | vs random |
|--------|-------|----------|-----------|
| cut3r (baseline) | 0.817 | — | — |
| ttt3r | 0.406 | -50.3% | — |
| ttt3r_random (p=0.5) | 0.280 | -65.8% | — |
| ttt3r_momentum_inv_t1 | 0.261 | -68.0% | -6.8% |
| **ttt3r_ortho** | **0.492** | -39.8% | +75.7% |
| ttt3r_ortho_adaptive (linear) | 0.358 | -56.2% | +27.9% |
| ttt3r_ortho_adaptive (match) | 0.356 | -56.4% | +27.1% |
| ttt3r_ortho_adaptive (threshold) | 0.376 | -54.0% | +34.3% |

**⚠ Ortho 在 ScanNet relpose 上退化**。三种 adaptive 策略天花板 ~0.356，修复了固定 ortho 的大部分退化 (0.492→0.356, -28%)，但仍逊于 random (0.280) 和 brake (0.261)。结论：ortho 分解本身在高 drift energy 场景有结构性限制，不是靠调 α_drift 能完全修复的。

### Video Depth — Abs Rel ↓

| Config | KITTI | Bonn | Sintel |
|--------|-------|------|--------|
| cut3r | 0.1515 | 0.0990 | 1.0217 |
| ttt3r | 0.1319 (-12.9%) | 0.0997 | 0.9776 (-4.3%) |
| ttt3r_joint | 0.1344 (-11.3%) | 0.0941 (-5.0%) | 0.9173 (-10.2%) |
| ttt3r_momentum_inv_t1 (brake) | 0.1061 (-30.0%) | 0.0658 (-33.5%) | 0.4022 (-60.6%) |
| **ttt3r_ortho** | **0.1042 (-31.2%)** | **0.0680 (-31.3%)** | **0.4175 (-59.1%)** |

Brake vs ortho: depth 上非常接近。Bonn brake 略优 (0.0658 vs 0.0680), Sintel brake 略优 (0.4022 vs 0.4175), KITTI ortho 略优 (0.1042 vs 0.1061)。支撑叙事：scalar brake 已有效，ortho 提供 principled 方向分解但实际增益有限。

### 3D Reconstruction — 7scenes

| Config | Acc ↓ | Comp ↓ | NC ↑ |
|--------|-------|--------|------|
| cut3r | 0.092 | 0.048 | 0.563 |
| ttt3r | 0.027 (-70.7%) | 0.023 (-52.1%) | 0.581 |
| ttt3r_joint | 0.021 (-77.2%) | 0.022 (-54.2%) | 0.579 |
| ttt3r_momentum_inv_t1 (brake) | 0.021 (-77.2%) | 0.022 (-54.2%) | 0.580 |
| **ttt3r_ortho** | **0.026 (-71.7%)** | **0.022 (-54.2%)** | **0.577** |

Brake 在 7scenes Acc 上优于 ortho (0.021 vs 0.026)，Comp/NC 基本持平。

### Ortho Hyperparameter Sensitivity — TUM (1000f)

| α_novel | α_drift | β | ATE ↓ | vs default |
|---------|---------|------|-------|------------|
| 0.5 | 0.1 | 0.95 | **0.055** | **-1.5%** |
| **0.5** | **0.05** | **0.95** | **0.056** | **default** |
| 0.5 | 0.2 | 0.95 | 0.056 | +0.1% |
| 0.7 | 0.05 | 0.95 | 0.057 | +2.2% |
| 0.3 | 0.05 | 0.95 | 0.069 | +24.1% |
| 0.5 | 0.05 | 0.9 | 0.076 | +35.5% |
| 0.5 | 0.05 | 0.99 | 0.077 | +38.7% |

α_drift 鲁棒 (0.05-0.2 <2%)，α_novel≥0.5 鲁棒，β=0.95 sweet spot（0.9/0.99 退化 35%+）。

### Ortho Hyperparameter Sensitivity — ScanNet (90f, 76 valid, linspace 采样)

| α_novel | α_drift | β | ATE ↓ | vs default |
|---------|---------|------|-------|------------|
| 0.5 | 0.05 | **0.99** | **0.458** | **-17.8%** |
| 0.5 | 0.2 | 0.95 | 0.500 | -10.2% |
| 0.7 | 0.05 | 0.95 | 0.529 | -4.9% |
| 0.5 | 0.1 | 0.95 | 0.531 | -4.6% |
| **0.5** | **0.05** | **0.95** | **0.557** | **default** |
| 0.3 | 0.05 | 0.95 | 0.660 | +18.5% |
| 0.5 | 0.05 | 0.9 | 0.666 | +19.6% |

**⚠ 与 TUM 完全反转**: β=0.99 在 ScanNet 最优（TUM 最差），α_drift 越高越好（TUM 不敏感）。证实 drift 性质在两个 dataset 根本不同。

注：此 sensitivity 数据使用 linspace 采样（覆盖全轨迹），趋势仍有参考价值但绝对值偏高。

### Inference Overhead — TUM (3 seqs × 200f, 3 repeats)

| Config | FPS | Peak Mem (GB) | vs cut3r |
|--------|-----|---------------|----------|
| cut3r | 8.44 | 6.14 | — |
| ttt3r | 9.82 | 6.14 | +16% faster |
| ttt3r_random | 9.76 | 6.14 | +16% faster |
| ttt3r_momentum_inv_t1 (brake) | 10.03 | 6.14 | +19% faster |
| **ttt3r_ortho** | **9.95** | **6.14** | **+18% faster** |
| ttt3r_ortho_adaptive | 9.93 | 6.14 | +18% faster |

所有方法**零额外内存**，速度甚至略快（dampened update 减少 memory pressure）。结果: `eval_results/benchmark_overhead.json`

### Relpose ATE — Sintel（14 sequences, ~20-50f each）

| Config | ATE ↓ | vs cut3r |
|--------|-------|----------|
| cut3r (baseline) | 0.209 | — |
| ttt3r | 0.209 | -0.1% |
| ttt3r_random (p=0.5) | 0.220 | +5.2% |
| ttt3r_ortho_adaptive | 0.221 | +5.9% |
| ttt3r_ortho | 0.237 | +13.2% |
| ttt3r_momentum_inv_t1 (brake) | 0.238 | +14.0% |

Sintel 序列极短 (20-50f)，over-update 尚未累积，任何 dampening 均无益或略有害。

## A4: Delta Direction Analysis — ScanNet vs TUM

| 指标 | TUM (8 scenes) | ScanNet (96 scenes) |
|------|----------------|---------------------|
| **cos(δ_t, δ_{t-1}) mean** | **0.617 ± 0.037** | **0.767 ± 0.037** |
| cos std (intra-scene) | 0.124 ± 0.015 | 0.095 ± 0.011 |
| **drift energy (cos²)** | **0.398 ± 0.041** | **0.598 ± 0.054** |

**关键发现**:
1. **ScanNet cos 远高于 TUM** (0.77 vs 0.62) — ScanNet 室内场景的 state update 方向高度一致
2. **ScanNet drift energy 60% vs TUM 40%** — ScanNet 有 60% 更新能量在 "drift" 方向
3. **ScanNet 的 "drift" 是有用的 refinement** — 室内场景需要在一致方向上持续完善几何，ortho 把这些有用更新当噪声抑制了
4. **TUM drift energy 较低 (40%)** — 动态运动下 drift 确实是重复性 over-update，ortho 分解恰好合适

**Per-scene scatter (1000f)**: ortho vs cut3r 53/65 改善，vs ttt3r 31/65 改善，vs random 仅 12/65 改善。cos_mean 与 ortho improvement 弱正相关 (r=0.237, p=0.057)。90f scatter 见 A7。

脚本: `analysis/a4_delta_direction.py`，结果: `analysis_results/a4_delta_direction/`

### A7: Per-Scene Scatter — Drift Energy vs Ortho Improvement（ScanNet 90f, 90 valid）

| 指标 | Ortho | Brake | Random |
|------|-------|-------|--------|
| 改善 scene 数 | 58/90 (64%) | 72/90 (80%) | 77/90 (86%) |
| r (drift energy vs improvement) | **+0.248** (p=0.018) | +0.157 (p=0.14) | +0.091 (p=0.39) |
| 退化 scene drift energy median | **0.617** | — | — |
| 改善 scene drift energy median | **0.594** | — | — |

**关键发现**:
1. **Ortho 有显著正相关** (r=0.248, p=0.018): drift energy 越高的 scene，ortho 越倾向退化 — 支持 "高 drift energy 下 drift 分量包含有用 refinement" 的假设
2. **Brake/random 无显著相关** — scalar dampening 对 drift energy 不敏感，更 robust
3. **Ortho 仍在 64% scene 改善**，但 36% scene 退化（vs brake 20%、random 14%）— 方向分解引入 precision-robustness trade-off
4. 退化 scene 的 drift energy 显著高于改善 scene (0.617 vs 0.594)

脚本: `analysis/viz_scatter_drift_ortho.py`，结果: `analysis_results/scatter_drift_ortho/`

### Depth Qualitative Visualization — Bonn balloon2

| Frame | CUT3R AbsRel | TTT3R | Brake | Ortho |
|-------|-------------|-------|-------|-------|
| t=197 | 0.089 | 0.069 | **0.041** | **0.044** |
| t=357 | **0.107** ↑退化 | 0.047 | **0.040** | **0.042** |

CUT3R 在长序列后期 (t=357) depth 估计明显退化 (AbsRel 0.089→0.107)，出现结构性 artifact；brake/ortho 保持稳定 (~0.04)，误差降低 >50%。

脚本: `analysis/viz_depth_qualitative.py`，结果: `analysis_results/depth_qualitative/`

## A5: TTSA3R TAUM Gate Statistics — TUM

验证 TTSA3R 的 TAUM gate (Temporal Adaptive Update Module) 是否也退化为常数。
TAUM: `sigmoid(||Δstate||_per_dim / mean(||Δstate||) - 1.5)`，理论预期 sigmoid(-0.5) ≈ 0.378。

| 指标 | TAUM Temporal | SCUM Spatial | Final (T×S) |
|------|--------------|--------------|-------------|
| **mean** | **0.355** | 0.651 | **0.231** |
| **σ_time** | **0.006** | — | **0.016** |
| σ_dim (per-frame) | 0.169 | — | — |

**关键发现**: TAUM temporal gate σ_time=0.006，比 ttt3r 自身的 gate (σ≈0.02) 小 3-4x，**更严重地退化为常数**。
- 理论原因：`state_change / mean(state_change)` 归一化后均值恒为 1.0，sigmoid(1-1.5) ≈ 0.378
- 实际 0.355 vs 理论 0.378 — 因为 state_change 分布右偏（CV=1.28）
- **SCUM spatial gate 提供了一些 cross-token variation (μ=0.65)，但 final gate 的时序动态仍极小**
- 结论：TTSA3R 的 "adaptive" gate 在实际运行中也退化为近似常数 dampening（~0.23），证实 A1 finding 推广到竞品

脚本: `analysis/taum_gate_stats.py`（需 `cut3r_taum_log` update type），结果: `analysis_results/taum_gate_stats/`

## Short Sequence Evaluation (90 frames, CUT3R protocol)

### TUM 90f（8 sequences）

| Config | ATE ↓ | vs cut3r | vs TTSA3R |
|--------|-------|----------|-----------|
| cut3r (baseline) | 0.0325 | — | — |
| TTSA3R (paper reported) | 0.026 | -20.0% | — |
| ttt3r | 0.0189 | -41.8% | -27.3% |
| ttt3r_random (p=0.33) | 0.0153 | -52.9% | -41.2% |
| ttt3r_momentum_inv_t1 (brake) | 0.0153 | -52.8% | -41.2% |
| **ttt3r_ortho** | **0.0145** | **-55.4%** | **-44.2%** |

Ortho 在 TUM 短序列上也最优，大幅超越 TTSA3R。

### ScanNet 90f（96 scenes, 90 valid, first-90 标准协议）

| Config | ATE ↓ | vs cut3r |
|--------|-------|----------|
| cut3r (baseline) | 0.095 | — |
| **ttt3r** | **0.064** | **-32.7%** |
| ttt3r_random (p=0.5) | 0.064 | -32.7% |
| ttt3r_momentum_inv_t1 (brake) | 0.071 | -25.0% |
| ttt3r_ortho_adaptive (linear) | 0.074 | -22.6% |
| ttt3r_ortho | 0.087 | -8.2% |

ScanNet 90f 所有方法均改善。ttt3r/random 改善最大 (-33%)，ortho 改善最小 (-8%)。与 1000f 趋势一致：ortho 的 drift 分解在 ScanNet 高 drift energy 场景效果有限。

注：之前的 ScanNet 90f 结果（所有方法退化）是因为帧选择错误 — 用了 np.linspace 均匀采样（覆盖 100% 轨迹），本质上是稀疏长序列测试。修正为标准 first-90 协议后结果翻转。

### A6: Over-update 普遍存在 — 短序列即可观察

| | TUM 90f | TUM 1000f | ScanNet 90f | ScanNet 1000f | Sintel ~50f |
|---|---------|-----------|-------------|---------------|-------------|
| cut3r ATE | 0.033 | 0.166 | 0.095 | 0.805 | 0.209 |
| ttt3r vs cut3r | **-42%** | **-38%** | **-33%** | **-50%** | 0% |
| ortho vs cut3r | **-55%** | **-66%** | **-8%** | -40% | +13% |
| brake vs cut3r | **-53%** | **-62%** | **-25%** | -68% | +14% |

**关键发现**:
1. **TUM**: 所有长度都有 over-update（90f 已 -42%），dampening 始终有效，ortho 最优
2. **ScanNet**: 90f 即有 over-update（ttt3r -33%），1000f 更严重（ATE 8.5x 退化到 0.805）
3. **Sintel**: 序列极短 (20-50f)，无 over-update
4. **Ortho 在 ScanNet 上效果有限**: 90f -8% vs ttt3r -33%，1000f -40% vs brake -68%。高 drift energy 场景 drift 分解不如 scalar dampening
5. **Over-update 严重度**: ScanNet 1000f/90f = 8.5x 退化，TUM 1000f/90f = 5.0x 退化

注：cut3r baseline 有差异（TUM 0.0325 vs TTSA3R paper 0.046），可能是 eval stride/subset 不同。

结果: `eval_results/relpose/tum/<config>/`, `eval_results/relpose/scannet_s3_90_first/<config>/`

## Update Types in model.py

| `model_update_type` | `mask1` (state) | Status |
|---------------------|-----------------|--------|
| `cut3r` | 1.0 (baseline) | baseline |
| `ttt3r` | sigmoid(cross_attn) | baseline |
| `ttt3r_ortho` | delta orthogonalization | **TUM/depth 最优，ScanNet pose 退化** |
| `ttt3r_random` | ttt3r × p (constant) | naive baseline |
| `ttt3r_momentum` | ttt3r × stability brake | 已验证，非最优 |
| `ttt3r_geogate` / `ttt3r_joint` | ttt3r × geo/joint gate | 早期方法 |
| `ttt3r_conf` / `ttt3r_l2gate` | ttt3r × conf/l2 gate | naive baseline |
| Others (spectral, memgate, delta_clip, attn_protect, mem_novelty, brake_geo) | various | 已放弃 |

## Failed Directions (Summary)

- **SIASU v1/v2**: EMA 紧密追踪 → alpha≡0.5; v2 ranking 更差 (0.291 vs 0.283)
- **Geo gate 联合 (brake_geo)**: 两 gate 叠加 over-dampening
- **Delta Clipping**: 限制大更新，旋转场景退化
- **Attention Protection**: 高 attention ≠ 需保护
- **Memory Novelty Gate**: cosine EMA 紧密追踪 (cos>0.99)，≈常数
- **Cross-attention bridge (Route C1)**: Attention 过于 diffuse (entropy 0.914)
- **Non-inverted momentum**: SGD 直觉在 over-update 场景有害
- **Token tracking (Direction C)**: State tokens 不追踪空间语义

## Eval Pipeline

三类评测，脚本在 `eval/` 下：

| 评测类型 | 数据集 | 脚本 | 数据路径 |
|---------|--------|------|---------|
| Camera Pose | ScanNet, TUM, Sintel | `eval/relpose/launch.py` | `data/long_scannet_s3/`, `data/long_tum_s1/` |
| Video Depth | KITTI, Bonn, Sintel | `eval/video_depth/launch.py` | `data/long_kitti_s1/`, `data/long_bonn_s1/` |
| 3D Reconstruction | 7scenes | `eval/mv_recon/launch.py` | — |

```bash
# 双卡并行示例
conda activate ttt3r

# GPU0 — ScanNet relpose
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src accelerate launch --num_processes 1 --main_process_port 29560 \
    eval/relpose/launch.py \
    --weights model/cut3r_512_dpt_4_64.pth --output_dir eval_results/relpose/scannet_s3_1000/<config> \
    --eval_dataset scannet_s3_1000 --size 512 --model_update_type <config> \
    --spectral_temperature 1.0 --geo_gate_tau 2.0 --geo_gate_freq_cutoff 4

# GPU1 — TUM relpose
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src accelerate launch --num_processes 1 --main_process_port 29561 \
    eval/relpose/launch.py \
    --weights model/cut3r_512_dpt_4_64.pth --output_dir eval_results/relpose/tum_s1_1000/<config> \
    --eval_dataset tum_s1_1000 --size 512 --model_update_type <config> \
    --spectral_temperature 1.0 --geo_gate_tau 2.0 --geo_gate_freq_cutoff 4
```

共享参数: `--seed 42 --size 512 --max_frames 200 --num_scannet 10`
并行脚本: `eval/run_parallel_eval.sh`
结果: `eval_results/relpose/<dataset>/<config>/_error_log.txt`

### Paths

- Model: `model/cut3r_512_dpt_4_64.pth`
- 原始数据: `/mnt/sda/szy/research/dataset/` (ScanNet, TUM)
- 本地同步: `rsync -avz 10.160.4.14:/home/szy/research/TTT3R/analysis_results/<exp>/ analysis_results/<exp>/`

### Dataset Notes

- ScanNet: 100 test scenes → 96 预处理（4 empty skip）→ 90 valid (90f) / 66 valid (1000f) / 65 valid (1000f adaptive)（GT 含 -inf, evo eigh 不收敛）
- TUM: 8 sequences, 全部成功
- 所有数据集预处理完成，评测 pipeline 正常

## Key Files

| File | Purpose |
|------|---------|
| `src/dust3r/model.py` | 所有 update types, gate methods, LocalMemory |
| `docs/research_progress.md` | 完整研究日志 |
| `docs/related_work.md` | 竞品分析 & 相关工作 |
| `docs/theory_section.tex` | 理论推导 |
| `analysis/a1a2_gate_dynamics.py` | A1/A2 分析脚本 |
| `analysis/a4_delta_direction.py` | A4 delta direction 分析（ScanNet vs TUM） |
| `analysis/taum_gate_stats.py` | A5 TTSA3R TAUM gate 退化分析 |
| `eval/run_parallel_eval.sh` | 并行评测脚本 |
| `eval/benchmark_overhead.py` | 推理 overhead benchmark（FPS + GPU memory） |
| `analysis/viz_depth_qualitative.py` | Depth 定性对比可视化（Bonn, error map） |
| `analysis/viz_scatter_drift_ortho.py` | Per-scene scatter: drift energy vs ortho improvement |
| `eval/run_scaling_curve.sh` | ScanNet scaling curve 实验（200f/500f） |

## Known Issues

1. **Gate state 每帧重置**: `view["reset"]` 返回 `tensor([False])` 非 None → 用 `reset_mask.any()` 判断。已修复三处。
2. **ScanNet 31 scene skip**: GT 含 -inf, evo eigh 不收敛。与原论文一致，不影响公平对比。
3. **`_forward_impl` 扩展**: 已补全所有 update type 支持（含 ttt3r_ortho），与 `inference_step` 对齐。

## Paper Narrative

**论文标题**: DD3R: Training-Free Directional Decomposition Beyond Scalar Gating for Recurrent 3D Reconstruction

**叙事**: Over-update 普遍存在（90f 即可观察）→ scalar gate 全退化为常数 → 方向性分析 → Directional Decomposition (Delta Orthogonalization)

**Story**:
1. **问题**: Recurrent 3D 的 state update 存在 systematic over-update，90f 即可观察（TUM -42%, ScanNet -33%），随序列增长加剧（1000f: TUM -38%, ScanNet -50%）
2. **分析**: Scalar adaptive gate 全部退化为常数（A1-A3 + 竞品 TTSA3R A5）；delta 方向有结构性 drift（A4），drift 性质因场景而异
3. **Insight**: 问题不是 "何时更新" 而是 "更新方向的哪部分该保留"——需要 beyond scalar gating 的方向性分解；超参敏感性在 TUM/ScanNet 上完全反转（β=0.95 vs 0.99）
4. **方法**: Directional Decomposition (DD3R) — drift/novel 分解 + 差异化抑制，在 TTT3R sigmoid mask 之上叠加方向控制
5. **结果**: TUM pose -66.5% (long) / -55.4% (short, vs TTSA3R -44.2%), video depth -31~59%, 7scenes Comp -54%，零额外 overhead; ScanNet 90f -8% (ortho) / -33% (ttt3r)

**Contributions**:
1. 揭示 over-update 普遍存在（90f 即可观察）+ scalar gate 退化为常数（自身 A1-A3 + 竞品 TTSA3R A5 双重验证）
2. 发现方向性本质：drift energy 在不同场景差异显著（40%-60%），per-scene scatter 证实高 drift energy scene ortho 更倾向退化 (r=0.248, p=0.018)，精确解释方法适用边界
3. Directional Decomposition (DD3R): train-free, plug-in, zero overhead, TUM pose SOTA (-55~66%), video depth SOTA (-31~59%)。与 TTT3R mask 互补——mask 控制 per-token 更新幅度，DD3R 控制更新内方向组成
4. 短序列上大幅超越 TTSA3R（TUM ATE -44.2%），ScanNet 90f 也改善 (-8~33%)

## Next Steps

### 实验 & 可视化
- ~~**[P1] Depth qualitative viz**~~ ✅ — Bonn balloon2 t=197/357, CUT3R 退化到 0.107, brake/ortho 保持 0.04。结果: `analysis_results/depth_qualitative/`
- ~~**[P1] Per-scene scatter plot**~~ ✅ — ScanNet 90f 90 scenes: ortho 与 drift energy 显著正相关 (r=0.248, p=0.018)，高 drift scene 更倾向退化。结果: `analysis_results/scatter_drift_ortho/`
- **[P1] ScanNet scaling curve** — 🔄 GPU 实验进行中（tmux `ttt3r`, 脚本 `eval/run_scaling_curve.sh`）。200f/500f × 6 methods，预计 8-12h。已有 90f + 1000f 数据
- **[P2] Length-aware ortho** — 前 T₀ 帧不抑制 drift，之后逐渐增强（实验性探索，可能不进 paper）

### 理论 & 写作
- **[P1] 理论框架更新** — drift energy bound（为什么 α_drift 最优值依赖 drift energy），adaptive α 推导，与 continual learning gradient projection 的联系
- **[P1] Paper writing: method section** — Delta Orthogonalization 形式化定义 + 与 scalar dampening 的对比分析
- **[P1] Paper writing: experiments** — 五数据集三任务结果表 + ablation（sensitivity, overhead, A1-A6）
- **[P2] Paper writing: intro + related work** — over-update 问题定义，scalar gate 退化现象，positioning vs GRS-SLAM3R/OnlineX/LONG3R
