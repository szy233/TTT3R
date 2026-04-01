# DDD3R — Directional Decomposition and Dampening for Recurrent 3D Reconstruction

## Project Goal
NeurIPS submission. **一个大问题 → 一个统一方案**。诊断 recurrent 3D reconstruction 中 state update 调节的系统性失效，提出 DDD3R 作为统一解决方案。

## Paper Narrative (Revised)

### 大问题（一句话）
**Recurrent 3D reconstruction 的 state update 缺乏有效调节，导致系统性退化。**

### 三个 Manifestation（递进关系）

**M1: 幅度失控。** Update 整体过大，90f 即可观察，随序列长度持续累积（ScanNet 8.5x, TUM 5.0x）。常数 dampening 即可改善 30-68%，证明 over-update 是 dominant failure mode。

**M2: 现有自适应 gate 全部失效。** TTT3R sigmoid gate σ≈0.02, TTSA3R TAUM σ≈0.006，均退化为近似常数。理论原因明确（sigmoid 饱和 / 归一化后均值恒为1）。这解释了为什么问题没被已有工作解决。

**M3: Update 方向包含结构性冗余。** Delta 中大量沿历史方向的 drift 分量，但性质因场景而异（TUM drift energy ~40% = 有害重复, ScanNet ~60% = 有用 refinement）。光控制幅度不够，还需要理解方向。

递进：M1"有多严重" → M2"为什么没被解决" → M3"更深层是什么"。

### 统一方案：DDD3R
一个 training-free、plug-in、zero-overhead 的 state update 调节框架。核心是一个统一 update rule，所有方法（constant dampening、temporal brake、directional decomposition）都是它的特殊情况，由参数 $(\alpha_\perp, \alpha_\parallel, \gamma)$ 控制在同一 spectrum 上的位置。

**统一公式：**
```
S_t = S_{t-1} + β_t (α⊥ · δ⊥ + α∥ · δ∥)
```
- α⊥ = α∥ → constant dampening（无方向感知，回应 M1）
- α⊥ > α∥, γ=0 → fixed directional decomposition（回应 M1+M3）
- α⊥ > α∥, γ>0 → drift-adaptive（自动在 ortho↔constant 间滑动，回应 M1+M3 + self-correction）
- Temporal brake → directional decomposition 的一阶近似（回应 M2，作为 baseline 展示）

### Contributions（两个）
**C1 诊断:** 揭示 recurrent 3D reconstruction 中 state update 调节的三重失效——幅度失控、自适应 gate 退化为常数、方向冗余——并证明 over-update 是 dominant failure mode。

**C2 统一方案:** 提出 DDD3R，training-free、zero-overhead 的统一调节框架。一个 update rule 统摄所有方法变体，drift energy 作为核心控制变量自动决定 operating point。TUM pose -55%/-66%，video depth -31%~-59%，零额外开销。

### Paper Structure
- **Section 1 Introduction**: 一个问题，一个方案，两个贡献
- **Section 3 Analysis**: 统一诊断叙事，M1→M2→M3 递进
- **Section 4 Method**: DDD3R 统一 update rule，三个 stage（decompose→reweight→gate）分别回应诊断发现，然后展示 continuous spectrum + drift-adaptive
- **Section 5 Experiments**: DDD3R 整体效果 + spectrum ablation 展示 γ 控制的连续滑动

---

## 路线决策：已确定 — 统一 Spectrum

### 最终路线：Continuous Spectrum（非叠加，非独立）

不走路线 A（叠加式模块）也不走路线 B（独立 operating points），而是**统一 spectrum**：所有方法都是同一个 update rule 的特殊情况，由参数控制在 spectrum 上的位置。

**核心叙事：** DDD3R 不是三个并列方法，而是一个参数化家族。Constant dampening 不是"框架外的 baseline"，而是 α⊥ = α∥ 时方向分解自动消失的退化。Brake 不是并列方法，而是 directional decomposition 的一阶近似。Steep adaptive (γ>0) 实现 ortho↔constant 的连续滑动。

**解决的两个关键矛盾：**

**矛盾 1：主推 ortho 但它不是全场景最优**
- 不把 ortho 包装成"最强方法"，而是"诊断驱动的方法设计范例"
- TUM 上 ortho 端最优 (0.056) → 当 drift 确实有害时，方向分解是对的
- ScanNet 上 steep 自动退化为 brake-like → framework 能 self-correct
- Constant 是 α⊥=α∥ 的 exact special case，不是"框架外的方法反而更好"

**矛盾 2：steep 证明 ortho 能自适应，但也在退化为 brake-like**
- 这恰恰说明 framework 的 self-correction 能力
- γ 控制 spectrum 位置，drift energy 是核心控制变量

**Spectrum 展示表格（experiments section, e^γ 公式）：**

| DDD3R ($\gamma$) | TUM 1000f | ScanNet 1000f | KITTI seq04 | KITTI seq03 | 行为 |
|---|---|---|---|---|---|
| α⊥=α∥=0.5 (constant) | 0.066 | 0.280 | 5.10 | 60.68 | 无方向感知 |
| brake (baseline) | 0.063 | 0.311 | 7.05 | 25.38 | 纯时序自适应 |
| γ=0.5 | 🔄 | 🔄 | 🔄 | 🔄 | 轻度 ortho |
| γ=1 | 🔄 | 🔄 | 🔄 | 🔄 | 适度 ortho |
| γ=2 | 0.056 | 0.394 | 8.78 | 24.58 | 较强 ortho |
| γ=3 | 0.054 | 0.407 | 8.14 | 26.79 | 强 ortho |
| γ=5 | 0.055 | 0.456 | 10.37 | 26.01 | 接近 pure ortho |
| γ=0 (pure ortho) | 0.056 | 0.492 | 17.79 | 39.91 | 全方向分解 |

**γ 方向性**：γ→∞ = pure ortho, γ→0 = isotropic。e^γ 是保守公式，在 TUM (e≈0.4) 上 γ=2~5 都保持 ortho 优势，ScanNet (e≈0.6) 上需要小 γ (≤2) 才能有效 self-correct。

**待验证**：γ=1 可能是 cross-dataset sweet spot（TUM e^1=0.4 仍保留方向分解，ScanNet e^1=0.6 有效推向 isotropic）。

---

## Architecture Overview

Model: `src/dust3r/model.py`, class `ARCroco3DStereo`. Recurrent processing:
1. Encode frame → `feat_i`
2. `_recurrent_rollout(state_feat, feat_i)` → `new_state_feat`, `dec`
3. `pose_retriever.update_mem(mem, feat, pose)` → `new_mem`
4. `_downstream_head(dec)` → `res` (pts3d, conf)
5. State update: `state_feat = new * mask1 + old * (1-mask1)`
6. Memory update: `mem = new_mem * mask2 + mem * (1-mask2)`

`mask1` is where our method is applied.

## Methods（统一 Spectrum）

**统一 update rule（所有方法都是特殊情况）：**
```
S_t = S_{t-1} + β_t (α⊥ · δ⊥ + α∥ · δ∥)
```

三个 stage：
1. **Decompose (→M3):** EMA 跟踪 drift direction，投影分解 δ = δ∥ + δ⊥
2. **Reweight (→M1):** α⊥ > α∥ 差异化抑制，drift 压得更狠
3. **Gate (→M2):** 保留 β_t 作为 per-token spatial mask（虽然退化为常数，但 spatial selectivity 仍有意义）

### 符号约定
- **α⊥**: orthogonal component 系数（保留 novel 信息），默认 0.5
- **α∥**: drift component 系数（抑制冗余），默认 0.05
- **β_ema**: EMA momentum，默认 0.95
- **γ**: steep exponent，控制 drift-adaptive 程度，默认 0（fixed decomposition）
- **d_t**: EMA drift direction（不做 normalize），`d_t = β_ema · d_{t-1} + (1-β_ema) · δ̂_t`
- **δ̂_t**: unit-normalized delta，`δ_t / ‖δ_t‖`
- **e_t**: instantaneous drift energy，`⟨δ̂_t, d̂_t⟩²`

### Spectrum 特殊情况

| 参数设置 | 等价方法 | 方向感知 |
|---------|---------|---------|
| α⊥ = α∥ = α | constant dampening | 无 |
| α⊥ > α∥, γ=0 | DDD3R (fixed ortho) | 全方向分解 |
| α⊥ > α∥, γ>0 | DDD3R (drift-adaptive) | 自适应 |

### CLI 参数对照

| Paper Symbol | CLI Arg | Config Attr | Default |
|-------------|---------|------------|---------|
| α (constant) | `--alpha` | `config.alpha` | 0.5 |
| α⊥ | `--alpha_perp` | `config.alpha_perp` | 0.5 |
| α∥ | `--alpha_parallel` | `config.alpha_parallel` | 0.05 |
| β_ema | `--beta_ema` | `config.beta_ema` | 0.95 |
| γ | `--gamma` | `config.gamma` | 0.0 |
| τ (brake) | `--brake_tau` | `config.brake_tau` | 2.0 |

Old CLI args (`--ortho_alpha_novel`, `--ortho_beta`, etc.) still work as hidden aliases.

### Brake 的定位
Brake 不作为 DDD3R 的组件，而是**作为 baseline 展示**。它是 directional decomposition 的一阶近似：
- Brake: cos(δ_t, δ_{t-1}) 高 → 整体 scale down（只看相邻帧，scalar gate）
- DDD3R: EMA 平滑 drift direction → 投影分解 → 差异化系数（多帧历史，directional gate）
Brake 在 Section 4 的 "Connection to temporal braking" paragraph 中讨论，在 Section 5 作为 baseline 出现。

### Steep Adaptive 公式
```
e_t = ⟨δ̂_t, d̂_t⟩²                              # per-token drift energy
w_t = e_t^γ                                       # weight toward isotropic
α∥^(t) = w_t · α⊥ + (1 - w_t) · α∥               # interpolate toward isotropic
```
- γ→0: w→1, α∥→α⊥ → 完全 isotropic（= constant dampening）
- γ→∞: w→0, α∥→α∥ → 完全 ortho（= fixed decomposition）
- e_t 高 (strong drift) → w 大 → 退化为 isotropic → 保留有用 drift
- e_t 低 (weak drift) → w 小 → 全方向分解 → 压制有害 drift

**注意**：γ 控制 ortho 保留程度，不是退化速度。γ 越大保留越多 ortho。

## Key Results

### Relpose ATE — Long Sequence (1000f)

**TUM（8 sequences）**

| Config | ATE ↓ | vs cut3r | 角色 |
|--------|-------|----------|------|
| cut3r (baseline) | 0.166 | — | |
| ttt3r | 0.103 | -38.1% | existing method |
| constant (p=0.33) | 0.066 | -60.2% | M1 severity evidence |
| brake | 0.063 | -61.8% | robust default |
| **ortho** | **0.056** | **-66.5%** | precision variant |

**ScanNet（96 scenes, 65 valid）**

| Config | ATE ↓ | vs cut3r | 角色 |
|--------|-------|----------|------|
| cut3r (baseline) | 0.817 | — | |
| ttt3r | 0.406 | -50.3% | existing method |
| constant (p=0.5) | 0.280 | -65.8% | M1 severity evidence |
| **brake** | **0.261** | **-68.0%** | robust default |
| ortho | 0.492 | -39.8% | ⚠ 退化（drift energy 高）|

**Sintel（14 sequences, ~20-50f）**— 序列极短，over-update 尚未累积，任何 dampening 均无益。

### Relpose ATE — Short Sequence (90f)

**TUM 90f**

| Config | ATE ↓ | vs cut3r | vs TTSA3R |
|--------|-------|----------|-----------|
| cut3r | 0.0325 | — | — |
| TTSA3R (paper) | 0.026 | -20.0% | — |
| ttt3r | 0.0189 | -41.8% | -27.3% |
| constant / brake | 0.0153 | -52.9% | -41.2% |
| **ortho** | **0.0145** | **-55.4%** | **-44.2%** |

**ScanNet 90f（96 scenes, 90 valid）**

| Config | ATE ↓ | vs cut3r |
|--------|-------|----------|
| cut3r | 0.095 | — |
| ttt3r / constant | 0.064 | -32.7% |
| brake | 0.071 | -25.0% |

### Video Depth — Abs Rel ↓

| Config | KITTI | Bonn | Sintel |
|--------|-------|------|--------|
| cut3r | 0.1515 | 0.0990 | 1.0217 |
| ttt3r | 0.1319 (-12.9%) | 0.0997 | 0.9776 (-4.3%) |
| brake | 0.1061 (-30.0%) | 0.0658 (-33.5%) | 0.4022 (-60.6%) |
| ortho | 0.1042 (-31.2%) | 0.0680 (-31.3%) | 0.4175 (-59.1%) |

### 3D Reconstruction — 7scenes

| Config | Acc ↓ | Comp ↓ | NC ↑ |
|--------|-------|--------|------|
| cut3r | 0.092 | 0.048 | 0.563 |
| ttt3r | 0.027 (-70.7%) | 0.023 (-52.1%) | 0.581 |
| brake | 0.021 (-77.2%) | 0.022 (-54.2%) | 0.580 |
| ortho | 0.026 (-71.7%) | 0.022 (-54.2%) | 0.577 |

### Inference Overhead — TUM (3 seqs × 200f, 3 repeats)

| Config | FPS | Peak Mem (GB) | vs cut3r |
|--------|-----|---------------|----------|
| cut3r | 8.44 | 6.14 | — |
| ttt3r | 9.82 | 6.14 | +16% faster |
| brake | 10.03 | 6.14 | +19% faster |
| ortho | 9.95 | 6.14 | +18% faster |

所有方法**零额外内存**，速度甚至略快。

## Analysis

### A1-A3: Scalar Gate 退化为常数（支撑 M2）
- **A1**: TTT3R 所有 scalar adaptive gate 退化为 ~constant 0.33（σ≈0.02），无时序动态
- **A2**: cos variance vs improvement 无相关（r=-0.13, p=0.63）→ adaptive timing 无意义
- **A3**: Scalar gate per-scene ~50/50 改善/退化 → 不 robust

### A4: Delta Direction — ScanNet vs TUM（支撑 M3）
| 指标 | TUM (8 scenes) | ScanNet (96 scenes) |
|------|----------------|---------------------|
| cos(δ_t, δ_{t-1}) mean | 0.617 ± 0.037 | 0.767 ± 0.037 |
| drift energy (cos²) | 0.398 ± 0.041 | 0.598 ± 0.054 |

ScanNet drift energy 60% vs TUM 40%。ScanNet 的 "drift" 是有用 refinement，ortho 误抑制；TUM 的 drift 是 over-update，ortho 恰好合适。

### A5: TTSA3R TAUM Gate 也退化为常数（支撑 M2）
TAUM temporal gate σ_time=0.006（比 ttt3r 的 σ≈0.02 小 3-4x），更严重退化为常数 ~0.355。
理论原因：`state_change / mean(state_change)` 归一化后均值恒为 1.0，sigmoid(1-1.5) ≈ 0.378。

### A6: Over-update 普遍存在（支撑 M1）
| | TUM 90f | TUM 1000f | ScanNet 90f | ScanNet 1000f | Sintel ~50f |
|---|---------|-----------|-------------|---------------|-------------|
| cut3r ATE | 0.033 | 0.166 | 0.095 | 0.805 | 0.209 |
| ttt3r vs cut3r | -42% | -38% | -33% | -50% | 0% |
| constant vs cut3r | -53% | -60% | -33% | -66% | +5% |
| brake vs cut3r | -53% | -62% | -25% | -68% | +14% |
| ortho vs cut3r | -55% | -66% | -8% | -40% | +13% |

Over-update 随长度加剧：ScanNet 1000f/90f = 8.5x，TUM = 5.0x。Sintel 极短无 over-update。

### A7: Per-Scene Scatter — Drift Energy vs Improvement（ScanNet 90f, 90 valid）
| 指标 | Ortho | Brake | Constant |
|------|-------|-------|----------|
| 改善 scene 数 | 58/90 (64%) | 72/90 (80%) | 77/90 (86%) |
| r (drift energy vs improvement) | +0.248 (p=0.018) | +0.157 (p=0.14) | +0.091 (p=0.39) |

Ortho 与 drift energy 显著正相关：drift energy 高 → 退化。Brake/constant 对 drift energy 不敏感。

### Qualitative
- **Depth** (Bonn balloon2): CUT3R 后期退化 (0.089→0.107)，brake/ortho 保持稳定 (~0.04)
- **Trajectory** (ScanNet 1000f): CUT3R 轨迹大幅漂移，brake/ortho 紧贴 GT（ATE 降低 85-90%）

### Ortho Hyperparameter Sensitivity
- **TUM**: α_drift 鲁棒 (0.05-0.2 <2%)，α_novel≥0.5 鲁棒，β=0.95 sweet spot
- **ScanNet**: 与 TUM 完全反转 — β=0.99 最优，α_drift 越高越好。证实 drift 性质根本不同。

### Steep Adaptive（e^γ 公式，已更新）

**公式更新**：从 `1-(1-e)^γ`（激进）改为 `e^γ`（保守）。γ 含义反转：γ→∞ = pure ortho, γ→0 = isotropic。

**TUM 1000f（e^γ 公式）：**

| γ | ATE ↓ | vs ortho | vs constant |
|---|-------|----------|-------------|
| 0 (pure ortho) | 0.056 | — | -15.6% |
| 2 | 0.056 | 0% | -15.6% |
| 3 | **0.054** | -3.6% | -18.2% |
| 5 | 0.055 | -1.4% | -16.7% |

保守公式在 TUM 上完美保持 ortho 优势（e≈0.4, e^2=0.16, 几乎不触发）。

**ScanNet 1000f（e^γ 公式）：**

| γ | ATE ↓ | vs ortho | vs constant |
|---|-------|----------|-------------|
| 0 (pure ortho) | 0.492 | — | +75.7% |
| 2 | 0.394 | -19.9% | +40.7% |
| 3 | 0.407 | -17.3% | +45.4% |
| 5 | 0.456 | -7.3% | +62.9% |

γ 增大 → 更 ortho → ScanNet 更差。需要 **小 γ**（γ=1 或 0.5）才能有效 self-correct。

**KITTI Odometry（室外验证）：**

| Config | seq 04 (271f) | seq 03 (801f) |
|---|---|---|
| cut3r | 21.85 | 150.29 |
| ttt3r | 11.06 | 103.26 |
| constant | **5.10** | 60.68 |
| brake | 7.05 | **25.38** |
| ortho (γ=0) | 17.79 | 39.91 |
| steep γ=2 | 8.78 | **24.58** |
| steep γ=3 | 8.14 | 26.79 |

Steep 大幅修复 ortho 退化（17.8→8.8 on seq04, 39.9→24.6 on seq03），seq03 接近 brake。

**进行中**：γ=1, γ=0.5 在 TUM/ScanNet/KITTI 上跑。γ=1 预期是 cross-dataset sweet spot。

## Update Types in model.py

| `model_update_type` | `mask1` (state) | Paper Role | Old Name |
|---------------------|-----------------|------------|----------|
| `cut3r` | 1.0 (baseline) | baseline | — |
| `ttt3r` | sigmoid(cross_attn) | existing method | — |
| `ddd3r_constant` | ttt3r × α | DDD3R with α⊥=α∥=α | `ttt3r_random` |
| `ddd3r_brake` | ttt3r × stability brake | baseline (DDD3R 一阶近似) | `ttt3r_momentum` |
| `ddd3r` | ttt3r_mask + directional decomposition | DDD3R with α⊥>α∥, γ≥0 | `ttt3r_ortho` |
| Others (joint, conf, l2gate, spectral, memgate, etc.) | various | abandoned | — |

Old names still work via backward-compat alias mapping in model.py.

## Eval Pipeline

| 评测类型 | 数据集 | 脚本 |
|---------|--------|------|
| Camera Pose | ScanNet, TUM, Sintel | `eval/relpose/launch.py` |
| Video Depth | KITTI, Bonn, Sintel | `eval/video_depth/launch.py` |
| 3D Reconstruction | 7scenes | `eval/mv_recon/launch.py` |

```bash
# Unified eval script (recommended)
bash eval/run_ddd3r_eval.sh <GPU> <DATASET> <METHOD>
# e.g.: bash eval/run_ddd3r_eval.sh 0 tum_s1_1000 ddd3r_g2

# Or direct launch:
conda activate ttt3r
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src accelerate launch --num_processes 1 --main_process_port 29560 \
    eval/relpose/launch.py \
    --weights model/cut3r_512_dpt_4_64.pth --output_dir eval_results/relpose/scannet_s3_1000/ddd3r \
    --eval_dataset scannet_s3_1000 --size 512 --model_update_type ddd3r --gamma 2.0
```

### Paths
- Model: `model/cut3r_512_dpt_4_64.pth`
- 原始数据: `/mnt/sda/szy/research/dataset/` (ScanNet, TUM)
- 本地同步: `rsync -avz 10.160.4.14:/home/szy/research/TTT3R/analysis_results/<exp>/ analysis_results/<exp>/`

### Dataset Notes
- ScanNet: 100 test scenes → 96 预处理（4 empty skip）→ 90 valid (90f) / 66 valid (1000f) / 65 valid (1000f adaptive)
- TUM: 8 sequences, 全部成功

## Key Files

| File | Purpose |
|------|---------|
| `src/dust3r/model.py` | 所有 update types, gate methods, LocalMemory |
| `docs/research_progress.md` | 完整研究日志 |
| `docs/related_work.md` | 竞品分析 & 相关工作 |
| `docs/theory_section.tex` | 理论推导 |
| `docs/method_section_v4.tex` | Method section 最终版（统一 spectrum 叙事）|
| `analysis/a1a2_gate_dynamics.py` | A1/A2 分析（支撑 M2）|
| `analysis/a4_delta_direction.py` | A4 delta direction 分析（支撑 M3）|
| `analysis/taum_gate_stats.py` | A5 TTSA3R TAUM gate 分析（支撑 M2）|
| `analysis/viz_depth_qualitative.py` | Depth 定性对比可视化 |
| `analysis/viz_scatter_drift_ortho.py` | Per-scene scatter: drift energy vs improvement |
| `analysis/viz_traj_comparison.py` | 轨迹对比可视化 |
| `eval/run_parallel_eval.sh` | 并行评测脚本 |
| `eval/benchmark_overhead.py` | 推理 overhead benchmark |
| `eval/run_scaling_curve.sh` | ScanNet scaling curve 实验 |
| `eval/run_steep_eval.sh` | Steep adaptive 实验（旧公式）|
| `eval/run_steep_v2.sh` | Steep v2 实验（e^γ 公式）|
| `eval/run_ddd3r_eval.sh` | 统一 DDD3R 评测脚本（paper naming）|

## Known Issues
1. **Gate state 每帧重置**: `view["reset"]` 返回 `tensor([False])` 非 None → 用 `reset_mask.any()` 判断。已修复。
2. **ScanNet scene skip**: GT 含 -inf, evo eigh 不收敛。与原论文一致，不影响公平对比。
3. **`_forward_impl` 扩展**: 已补全所有 update type 支持，与 `inference_step` 对齐。

## Next Steps

### 路线已确定：统一 Spectrum
- ✅ 路线决策完成 — 不走叠加也不走独立，走统一 spectrum
- ✅ Method section v4 完成 — `method_section_v4.tex`

### 已完成实验
- ✅ **路线 A 叠加实验** — constant+brake 在 TUM 上 mean ATE=0.099（退化，乘法式过度抑制）。路线 A 不可行。
- ✅ **Steep 公式更新** — 从 `1-(1-e)^γ` 改为 `e^γ`（论文公式）。γ 含义反转。
- ✅ **Steep v2 γ=2/3/5** — TUM + ScanNet + KITTI 全部完成
- ✅ **KITTI Odometry** — seq 03/04 baseline + steep 完成

### 进行中实验
- **[P0] Steep v2 γ=1, γ=0.5** — 🔄 TUM + ScanNet + KITTI 跑中。γ=1 是预期 sweet spot。
- **[P1] ScanNet scaling curve** — 🔄 GPU 进行中，200f/500f × 6 methods

### 写作
- ✅ **Method section** — v4 完成（统一 spectrum 叙事）
- **[P1] Analysis section** — M1→M2→M3 递进叙事
- **[P1] Experiments** — 五数据集三任务结果表 + spectrum ablation (γ 连续滑动)
- **[P2] Intro + related work** — positioning

### Method Section 写作决策记录
- **结构**: Preliminaries → 先给 boxed 总公式 → 三个 Stage 展开 → Continuous Spectrum → Implementation with Algorithm
- **符号**: α∥ (替代 α_drift), α⊥, n×c, Δ_t (大写 matrix), δ_t (per-token), d_t (EMA 不 normalize, projection 带 ‖d_t‖²)
- **Constant 定位**: α⊥=α∥ 的 exact special case，不在 Table 1 中单独出现
- **Brake 定位**: directional decomposition 的一阶近似，作为 baseline 展示，不作为 DDD3R 组件
- **Steep**: 在 Spectrum subsection 的 paragraph 中介绍，γ 控制 ortho↔constant 连续滑动
- **默认配置**: (α⊥, α∥, β_ema, γ) = (0.5, 0.05, 0.95, 0)，γ>0 作为 ablation
- **Steep 公式**: `w = e^γ`（保守，已对齐代码）。γ→0 isotropic, γ→∞ pure ortho。预期最优 γ≈1
