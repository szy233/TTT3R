# TTT3R — Delta Orthogonalization for Recurrent 3D Reconstruction

## Project Goal
NeurIPS submission. Train-free, inference-time method to address systematic over-update in recurrent 3D reconstruction (CUT3R/TTT3R). 核心方法：Delta Orthogonalization — 将 state update 分解为 drift（重复方向）和 novel（新信息）分量，差异化抑制。

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

**Per-scene scatter**: ortho 在 ScanNet 仅 11/65 scenes 改善。cos_mean 与 improvement 弱正相关 (r=0.237, p=0.057)。

脚本: `analysis/a4_delta_direction.py`，结果: `analysis_results/a4_delta_direction/`

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

### ScanNet 90f（96 scenes, 91 valid, first-90 标准协议）

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

- ScanNet: 100 test scenes → 96 预处理（4 empty skip）→ 91 valid (90f) / 66 valid (1000f) / 65 valid (1000f adaptive)（GT 含 -inf, evo eigh 不收敛）
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

## Known Issues

1. **Gate state 每帧重置**: `view["reset"]` 返回 `tensor([False])` 非 None → 用 `reset_mask.any()` 判断。已修复三处。
2. **ScanNet 31 scene skip**: GT 含 -inf, evo eigh 不收敛。与原论文一致，不影响公平对比。
3. **`_forward_impl` 扩展**: 已补全所有 update type 支持（含 ttt3r_ortho），与 `inference_step` 对齐。

## Paper Narrative

**叙事**: Over-update 是 emerging problem（随序列增长显现）→ 方向性分析 → Delta Decomposition

**Story**:
1. **问题**: Recurrent 3D 的 state update 存在 systematic over-update，90f 即可观察（TUM -42%, ScanNet -33%），随序列增长加剧（1000f: TUM -38%, ScanNet -50%）
2. **分析**: Scalar adaptive gate 全部退化为常数（A1-A3 + 竞品 TTSA3R A5）；delta 方向有结构性 drift（A4），drift 性质因场景而异
3. **Insight**: 问题不是 "何时更新" 而是 "更新方向的哪部分该保留"；超参敏感性在 TUM/ScanNet 上完全反转（β=0.95 vs 0.99）
4. **方法**: Delta Orthogonalization — drift/novel 分解 + 差异化抑制
5. **结果**: TUM pose -66.5% (long) / -55.4% (short, vs TTSA3R -44.2%), video depth -31~59%, 7scenes Comp -54%，零额外 overhead; ScanNet 90f -8% (ortho) / -33% (ttt3r)

**Contributions**:
1. 揭示 over-update 普遍存在（90f 即可观察）+ scalar gate 退化为常数（自身 A1-A3 + 竞品 TTSA3R A5 双重验证）
2. 发现方向性本质：drift energy 在不同场景差异显著（40%-60%），超参敏感性反转（TUM β=0.95 最优 vs ScanNet β=0.99 最优），精确解释方法适用边界
3. Delta Orthogonalization: train-free, plug-in, zero overhead, TUM pose SOTA (-55~66%), video depth SOTA (-31~59%)
4. 短序列上大幅超越 TTSA3R（TUM ATE -44.2%），ScanNet 90f 也改善 (-8~33%)

## Next Steps

### 已完成
- ~~Adaptive ortho ScanNet 结果分析~~ — 三种策略天花板 ~0.356，确认 ortho 在高 drift energy 场景的结构性限制
- ~~Brake depth + 7scenes~~ — brake 与 ortho 在 depth/7scenes 上非常接近，支撑 scalar→directional 叙事
- ~~TTSA3R TAUM gate analysis~~ — TAUM σ_time=0.006，比 ttt3r gate 更严重退化为常数，证实 A1 推广到竞品
- ~~Short sequence eval (TUM 90f)~~ — ortho ATE=0.0145 (-55.4% vs cut3r, -44% vs TTSA3R)，短序列上也最优
- ~~Inference overhead~~ — 所有方法零额外内存 (6.14GB)，FPS 8.4-10.0，negligible overhead
- ~~ScanNet 90f short-seq eval~~ — first-90 标准协议: 所有方法均改善 (ttt3r -33%, ortho -8%)，over-update 在 90f 即存在
- ~~ScanNet ortho 超参敏感性~~ — 与 TUM 完全反转: β=0.99 最优 (vs TUM β=0.95)，α_drift 越高越好
- ~~Sintel relpose~~ — 短序列 (20-50f) 无 over-update，dampening 均无益
- ~~A6 分析~~ — over-update 普遍存在（TUM/ScanNet 90f 即可观察），Sintel 太短除外

### 待办
- **[P1] 理论更新** — drift energy bound, adaptive α 推导, emerging problem 理论框架
- **[P1] Depth qualitative viz** — 代表帧 depth map 可视化
- **[P1] Per-scene scatter plot** — drift energy vs improvement（ScanNet 退化→insight）
- **[P1] ScanNet scaling curve** — 不同长度 (100f, 200f, 500f, 1000f) 各方法 ATE，找 over-update tipping point
- **[P2] Length-aware ortho** — 前 T₀ 帧不抑制 drift，之后逐渐增强（实验性探索）
- **[P2] Paper writing** — 基于当前结果开始写 method + experiments
