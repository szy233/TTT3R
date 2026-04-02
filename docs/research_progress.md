# TTT3R — 研究进展全记录

> 最后更新：2026-04-02

## 项目概述

**目标**：NeurIPS 投稿。针对循环式 3D 重建（CUT3R/TTT3R）的 **state over-update** 问题（90f 即可观察，非仅 long video），提出 train-free、inference-time 的方向性更新分解方案（Delta Orthogonalization）。

**代码库**：基于 CUT3R/TTT3R 框架，核心修改集中在 `src/dust3r/model.py`。

---

## 一、问题定义

CUT3R/TTT3R 用 recurrent state 处理视频帧：

```
state_t = new_state * mask + old_state * (1 - mask)
```

- `mask` 由 decoder cross-attention 学得（TTT3R），或恒为 1（CUT3R）
- 模型在 **image pairs** 上训练，但推理时在 **数百帧 long video** 上运行
- 每帧都以相同力度更新 state → **systematic over-update**

**核心证据**：任何 constant dampening（将 mask 乘以常数 0.5 或 0.33）都能显著改善，说明 over-update 是真实存在的核心瓶颈。

---

## 二、研究路线演变

```
频域三层框架 → Stability Brake → Scale Calibration 发现 → Delta 正交化
   (Phase 1)       (Phase 2)         (Phase 3)            (Phase 4, 当前)
```

### Phase 1：频域三层框架（已完成，部分放弃）

最初假设频域信号能在三个粒度上指导更新：

| Layer | 级别 | 信号 | 结果 | 状态 |
|-------|------|------|------|------|
| Layer 1 | 帧级 | LFE(FFT2(RGB_diff)) | Skip 35%帧, depth -3.1% | ✅ 有效但与细粒度冲突 |
| Layer 2 | Token级 | SIASU spectral energy | alpha ≡ 0.5, 无方差 | ❌ 放弃 |
| Layer 3 | State级 | LFE(FFT2(log_depth_diff)) | cut3r -3.5%, ttt3r -7.2% | ✅ 有效 |

**Joint Ablation** (10个ScanNet+TUM scene):

| 配置 | 误差 | vs cut3r |
|------|------|----------|
| cut3r (baseline) | 0.0745 | — |
| ttt3r | 0.0697 | -6.4% |
| **L23+ttt3r** | **0.0690** | **-7.5%** |
| L123+ttt3r | 0.0699 | -6.2%（L1冲突） |

**关键发现**：L2 (SIASU) 的 -8.2% 提升实际全部来自 alpha ≡ 0.5（常数乘法），非频域信号。

### Phase 2：Stability Brake（已完成，已验证）

**方法**：`α_t = σ(-τ·cos(δ_t, δ_{t-1}))`

- State 更新方向一致时制动（cos高→α小），突变时放行（cos低→α大）
- τ=1 最优

**全量 Relpose 结果 — ScanNet（96 scenes, 65 valid）**

| Config | ATE ↓ | vs cut3r | vs ttt3r | vs random(p=0.5) |
|--------|-------|----------|----------|-----------|
| cut3r | 0.817 | — | — | — |
| ttt3r | 0.406 | -50.3% | — | — |
| ttt3r_random (p=0.5) | 0.280 | -65.8% | -31.1% | — |
| **ttt3r_momentum_inv_t1** | **0.261** | **-68.0%** | **-35.6%** | **-6.8%** |

**全量 Relpose 结果 — TUM（8 sequences）**

| Config | ATE ↓ | vs cut3r | vs ttt3r | vs random(p=0.5) |
|--------|-------|----------|----------|-----------|
| cut3r | 0.166 | — | — | — |
| ttt3r | 0.103 | -38.1% | — | — |
| ttt3r_random (p=0.5) | 0.079 | -52.4% | -23.3% | — |
| ttt3r_random (p=0.33) | 0.066 | -60.2% | -35.9% | -16.5% |
| **ttt3r_momentum_inv_t1** | **0.063** | **-61.8%** | **-38.2%** | **-20.3%** |

### Phase 3：Scale Calibration 发现（关键转折点）

**发现 1：Stability brake ≈ 常数 ~0.33**

TUM 上 constant 0.33 的 ATE = 0.066，stability brake = 0.063，差距仅 **4.5%**。

**发现 2：A1 Gate Dynamics 分析确认**

| 场景 | cos 均值 | gate 均值 | gate std | 与 camera motion 相关性 |
|------|---------|-----------|----------|----------------------|
| TUM 静态 | 0.590 | 0.357 | **0.032** | -0.071 |
| TUM 行走 | 0.604 | 0.354 | **0.028** | -0.058 |
| ScanNet | 0.804 | 0.309 | **0.022** | +0.032 |

- Gate 标准差仅 0.02~0.03 → 事实上是常数 ~0.33
- 与 camera motion 相关性 ≈ 0 → "该刹时刹"的假设不成立
- Sigmoid 压缩了余弦信号的方差

**发现 3：A2 Var(cos) 与改善无相关性**

- ScanNet: Pearson r = -0.133, p = 0.289（不显著）
- TUM: Pearson r = -0.008（完全为零）

**发现 4：A3 Per-scene 一致性较弱**

- ScanNet: 31/65 scenes 改善（47.7%），mean +0.9%, std 23.4%
- TUM vs constant 0.33: 4/8 scenes 改善（50%）

**结论**：Over-update 的核心问题是 **scale mis-calibration**（更新总量过大），而非 timing/adaptiveness 问题。常数 dampening 解决了 95% 的问题。

### Phase 4：Delta 正交化（当前方向，突破性结果）

**核心洞察**：A1 分析显示 cos 均值 = 0.7 → state 的更新方向有 **70% 是重复的方向性漂移**。

**方法**：不是缩小更新量（constant dampening），而是**分解更新方向**：

```python
delta = new_state - old_state
drift_dir = normalize(EMA(delta))          # 一直在走的方向
drift_comp = proj(delta, drift_dir)        # 重复方向分量
novel_comp = delta - drift_comp            # 垂直于漂移的新信息分量

state = old + α_novel × novel_comp + α_drift × drift_comp
#       保留新信息(0.5)          压制漂移(0.05)
```

**为什么更有原则**：
- Constant dampening 对所有方向一视同仁 → 新信息也被压制
- Delta 正交化保留新信息方向（novel），只压制重复漂移（drift）
- 与 continual learning 的 gradient orthogonalization 有理论联系

---

## 三、全量 Baseline 对比（TUM Relpose）

以下为 TUM 上所有已测试方法的完整对比，按 ATE 排序：

| Config | ATE ↓ | vs cut3r | vs const 0.33 | 备注 |
|--------|-------|----------|--------------|------|
| cut3r | 0.166 | — | — | baseline |
| ttt3r | 0.103 | -38.1% | — | learned gate |
| ttt3r_momentum_v2 (non-inv) | 0.098 | -40.8% | — | 原始momentum, 有害 |
| ttt3r_random (p=0.5) | 0.079 | -52.4% | — | 常数 ×0.5 |
| ttt3r_novelty | 0.078 | -53.0% | +18.2% | 特征新颖度gate ❌ |
| ttt3r_centered | 0.077 | -53.6% | +16.7% | 居中gate ❌ |
| ttt3r_l2gate_fixed | 0.075 | -54.7% | +13.6% | L2范数gate ❌ |
| ttt3r_conf | 0.073 | -56.1% | +10.6% | 置信度gate ❌ |
| ttt3r_attn_protect | 0.070 | -57.8% | +6.5% | 注意力保护 ❌ |
| ttt3r_mem_novelty | 0.066 | -60.2% | +0.8% | 记忆新颖度 ≈ 常数 |
| ttt3r_random (p=0.33) | **0.066** | -60.2% | **—** | **常数 ×0.33** |
| ttt3r_momentum_inv_t1 | 0.063 | -61.8% | -3.8% | stability brake |
| **ttt3r_ortho (an=0.5, ad=0.05)** | **0.056** | **-66.3%** | **-15.4%** | **Delta 正交化 ✅** |
| ttt3r_delta_clip | 0.104 | -37.3% | +57.5% | delta剪裁 ❌ |
| ttt3r_true_momentum | 0.151-0.163 | ~-2% | — | 真momentum ❌ |

### ttt3r_ortho Per-Sequence 明细

| 序列 | const 0.33 | brake | **ortho** | ortho vs const |
|------|-----------|-------|-----------|---------------|
| sitting_halfsphere | 0.099 | 0.076 | **0.070** | -29.3% |
| sitting_rpy | 0.055 | 0.055 | **0.047** | -14.5% |
| sitting_static | 0.023 | 0.022 | **0.016** | -30.4% |
| sitting_xyz | 0.047 | 0.043 | **0.036** | -23.4% |
| walking_halfsphere | 0.088 | 0.086 | 0.086 | -2.3% |
| walking_rpy | 0.121 | 0.129 | **0.110** | -9.1% |
| walking_static | 0.019 | 0.019 | **0.018** | -5.3% |
| walking_xyz | 0.076 | 0.076 | **0.063** | -17.1% |
| **Average** | **0.066** | **0.063** | **0.056** | **-15.4%** |

**8 个序列中 7 个改善，1 个持平。**

---

## 四、正式评测结果（Stability Brake 之前的最终方法 ttt3r_joint）

以下为 ttt3r_joint (SIASU × GeoGate × TTT3R) 的正式评测结果。后续需替换为最新方法重新评测。

### Relpose ATE

| Config | ScanNet ↓ | TUM ↓ |
|--------|-----------|-------|
| cut3r | 0.671 (median) | 0.164 |
| ttt3r | 0.352 (-47.6%) | 0.104 (-36.4%) |
| ttt3r_joint | **0.214 (-68.1%)** | **0.059 (-64.1%)** |

### Video Depth (Abs Rel ↓)

| Config | KITTI | Bonn | Sintel |
|--------|-------|------|--------|
| cut3r | 0.1515 | 0.0990 | 1.0217 |
| ttt3r | 0.1319 (-12.9%) | 0.0997 | 0.9776 (-4.3%) |
| ttt3r_joint | **0.1344 (-11.3%)** | **0.0941 (-5.0%)** | **0.9173 (-10.2%)** |

### 3D Reconstruction — 7scenes

| Config | Acc ↓ | Comp ↓ | NC ↑ |
|--------|-------|--------|------|
| cut3r | 0.092 | 0.048 | 0.563 |
| ttt3r | 0.027 (-70.7%) | 0.023 (-52.1%) | 0.581 (+3.2%) |
| ttt3r_joint | **0.021 (-77.2%)** | **0.022 (-54.2%)** | 0.579 (+2.8%) |

---

## 五、Deep Analysis 结果

### A1：Gate Temporal Dynamics

**脚本**：`analysis/a1a2_gate_dynamics.py --mode a1`
**输出**：`analysis_results/a1a2_dynamics/a1_*.png`

每帧 cos(δ_t, δ_{t-1})、gate α_t、GT camera motion 三条曲线。覆盖 ScanNet (3 scenes)、TUM (2 scenes)、KITTI (3 scenes)、Sintel (4 scenes)。

**关键发现**：
- cos 均值 0.6~0.8 → state 持续沿同一方向漂移（over-update 确认）
- gate std 0.02~0.03 → sigmoid 压缩后 gate 几乎不变
- gate 与 camera motion 无相关（r ≈ 0）

### A2：Cosine Variance ↔ Improvement

**脚本**：`analysis/a1a2_gate_dynamics.py --mode a2`
**输出**：`analysis_results/a1a2_dynamics/a2_*.png`

- ScanNet (65 scenes): Pearson r = -0.133, p = 0.289
- TUM (8 scenes): Pearson r = -0.008, p = 0.986

**结论**：Var(cos) 无法预测 per-scene 改善幅度。"高运动多样性 → 更多改善"的假设不成立。

### A3：Per-Scene 改善分布

**脚本**：`analysis/a3_per_scene_distribution.py`
**输出**：`analysis_results/a3_per_scene/`

- ScanNet brake vs const 0.5: 31/65 改善 (47.7%), mean +0.9%, std 23.4%
- TUM brake vs const 0.33: 4/8 改善 (50%), mean +3.3%
- Best case: scene0793 +59.1%; Worst case: scene0789 -63.3%

**结论**：Stability brake 在均值上优于常数，但 per-scene 一致性较弱。

### Per-Token Gate 方差分析

**脚本**：`analysis/token_gate_variance.py`

- Token-level cosine std = 0.22（有变异）
- 但经 sigmoid 后 gate std = 0.05（被压缩为近似常数）
- Token-level spatial selectivity 无法通过 sigmoid 乘法传递

---

## 六、失败方向汇总

| 方向 | 信号 | TUM ATE | 失败原因 |
|------|------|---------|---------|
| SIASU v1 | per-token spectral energy | 0.283 (≡const×0.5) | EMA γ=0.95 紧密追踪 → alpha ≡ 0.5 |
| SIASU v2 | cross-token ranking | 0.291 | 排名信号无区分度 |
| Confidence gate | prediction confidence | 0.073 | 正反馈循环 |
| Geo + Brake 联合 | momentum × geo | 0.081 | 两gate叠加 over-dampening |
| Non-inverted momentum | cos高→更新多 | 0.098 | SGD直觉在over-update场景有害 |
| Feature novelty | EMA of feat cosine | 0.078 | EMA追踪过快, novelty≈0 |
| Centered gate | deviation from 0.5 | 0.077 | sigmoid压缩方差 |
| True momentum | EMA of deltas | 0.151-0.163 | 改变state分布, decoder崩溃 |
| Delta clipping | clip large delta norms | 0.104 | 旋转场景需要大delta |
| Attention protection | EWC-like importance | 0.070 | 高注意力token保护假设不成立 |
| Memory novelty | cosine vs EMA in proj_q space | 0.066 (≡const) | EMA追踪过快, novelty≈0 |
| Cross-attention bridge | pixel→token gate | — | attention太diffuse (entropy 0.914) |
| Dynamic token tracking | spatial token tracking | — | token不追踪空间语义 |

**规律**：所有 token-level spatial 方法都失败（sigmoid 压缩）。所有 EMA-based 方法都退化为常数。只有 delta 方向分解（ortho）跳出了这个困局。

---

## 七、关键 Bug 修复

1. **Gate state 每帧重置** (Critical, 2026-03-26): `if reset_mask is not None:` 永远 True → momentum/l2/spectral state 每帧重置。修复为 `if reset_mask.any():`。之前的 momentum 结果实际 = random。
2. **SIASU warm-start**: running_energy 初始化 0 → ratio 爆炸。修复为首帧 warm-start。
3. **`_forward_impl` 缺少扩展 update type**: 导致 mv_recon 评测失败。已补全。
4. **ScanNet pose 截断**: 根分区满导致 pose 文件被截断。已重新生成。
5. **ScanNet 31 scene Eigenvalue failure**: GT pose 含 -inf，三配置一致 skip，不影响对比。

---

## 八、当前状态与下一步

### 已完成（2026-03-29 更新）

- ✅ Ortho ScanNet/TUM/Sintel 全量评测（relpose + video depth + 7scenes）
- ✅ Ortho 超参敏感性（TUM + ScanNet，发现两者最优参数完全反转）
- ✅ Adaptive ortho 三种策略（linear/match/threshold），天花板 ~0.356
- ✅ Inference overhead benchmark（零额外内存，FPS +18%）
- ✅ A4 delta direction 分析（ScanNet drift energy 60% vs TUM 40%）
- ✅ A5 TTSA3R TAUM gate 退化分析（σ_time=0.006，比 ttt3r 更严重）
- ✅ A6 over-update 普遍性分析（90f 即可观察，非 emerging problem）
- ✅ ScanNet 90f 标准协议修正（linspace→first-90，结果翻转：退化→改善）
- ✅ Sintel relpose（短序列无 over-update，dampening 均无益）

### 待完成

| 优先级 | 任务 |
|--------|------|
| **P1** | 理论更新 — drift energy bound, adaptive α 推导, emerging problem 理论框架 |
| **P1** | Depth qualitative viz — 代表帧 depth map 可视化 |
| **P1** | Per-scene scatter plot — drift energy vs improvement（ScanNet 退化 → insight） |
| **P1** | ScanNet scaling curve — 不同长度 (100f, 200f, 500f, 1000f) 各方法 ATE |
| **P2** | Length-aware ortho — 前 T₀ 帧不抑制 drift，之后逐渐增强 |
| **P2** | Paper writing — method + experiments |

### Paper 叙事方向（2026-03-29 更新）

**核心叙事**：Over-update 是普遍存在的问题（90f 即可观察） → scalar gate 全退化为常数 → 方向性分析揭示 drift 本质 → Delta Orthogonalization

> **问题**：Recurrent 3D 的 state update 存在 systematic over-update，90f 即可观察（TUM -42%, ScanNet -33%），随序列增长加剧
>
> **分析**：Scalar adaptive gate 全部退化为常数（A1-A3 + 竞品 TTSA3R A5）；delta 方向有结构性 drift（A4），drift 性质因场景而异（TUM drift energy 40% vs ScanNet 60%）
>
> **Insight**：问题不是 "何时更新" 而是 "更新方向的哪部分该保留"；超参敏感性在 TUM/ScanNet 上完全反转（β=0.95 vs 0.99）
>
> **方法**：Delta Orthogonalization — drift/novel 分解 + 差异化抑制
>
> **结果**：TUM pose -66.5% (long) / -55.4% (short, vs TTSA3R -44.2%), video depth -31~59%, 7scenes Comp -54%，零额外 overhead; ScanNet 90f -8% (ortho) / -33% (ttt3r)
>
> **贡献**：(1) 揭示 over-update 普遍存在 + scalar gate 退化（双重验证）; (2) 发现方向性本质与 dataset-dependent drift; (3) Delta Orthogonalization: train-free, plug-in, zero overhead; (4) TUM/depth SOTA, 短序列超越 TTSA3R

### Phase 5：Attention Entropy Adaptive（2026-04-02 ~）

**动机**：Phase 4 的 auto-gamma 方案（warmup / steep_sigmoid / steep_clamp）都依赖 drift energy ē 来调节 γ 或 α_∥，但 drift energy 本身需要 EMA 方向估计收敛后才可靠，而且其物理含义("delta 与 drift 方向的对齐度")不直接对应"scene 是否在变化"。

**核心观察**：Cross-attention 每步已经算好 attention distribution（state 查 image），其 entropy 是一个 **zero-cost** 的场景变化指示器：
- Entropy 高 → attention 分散在多个 key → scene 在变化 / 新信息多 → drift 可能有用，少 suppress
- Entropy 低 → attention 集中在少数 key → 已收敛 / 信息饱和 → drift 有害，多 suppress

**方案**：
```
h_t = mean_over_layers_heads( H(softmax(attn_logits)) / log(N_keys) )   ∈ [0, 1]
h̄_t = β_h · h̄_{t-1} + (1-β_h) · h_t                                   (EMA smooth)
α_∥^(t) = h̄_t · α_⊥ + (1-h̄_t) · α_∥
```
- h̄→1 (高 entropy, scene 变化) → α_∥ → α_⊥ = 0.5（接近 isotropic，像 ScanNet）
- h̄→0 (低 entropy, 已收敛) → α_∥ → 0.05（aggressive decomposition，像 TUM）

**优势 vs auto-gamma**：
1. 完全去掉 γ 超参
2. 信号来自 attention（已有的计算），不依赖 drift energy 的收敛
3. 物理含义更直接：entropy 直接度量"当前帧带来多少新信息"
4. 实现为 `--auto_gamma entropy`，CLI: `--entropy_ema_beta 0.95`

**状态**：⬜ 待验证（TUM + ScanNet 1000f）

---

## 九、代码文件索引

| 文件 | 用途 |
|------|------|
| `src/dust3r/model.py` | 核心模型，所有 update type 和 gate 方法 |
| `eval/relpose/launch.py` | Relpose 评测脚本 |
| `eval/video_depth/launch.py` | Video depth 评测 |
| `eval/mv_recon/launch.py` | 3D reconstruction 评测 |
| `analysis/a1a2_gate_dynamics.py` | A1 gate dynamics + A2 variance correlation |
| `analysis/a3_per_scene_distribution.py` | A3 per-scene scatter/boxplot |
| `analysis/token_gate_variance.py` | Token-level gate 方差分析 |
| `analysis/s4_gate_visualization.py` | S4 gate activation 可视化 |
| `docs/theory_section.tex` | 理论推导 |
| `docs/related_work.md` | 相关工作整理 |

## 十、数据集与评测

| 数据集 | 原始路径 | 预处理 | 评测状态 |
|--------|---------|--------|---------|
| ScanNet | `/mnt/sda/szy/research/dataset/scannetv2` | `data/long_scannet_s3/` (96 scenes) | ✅ 完成 |
| TUM | `/mnt/sda/szy/research/dataset/tum` | `data/long_tum_s1/` (8 seqs) | ✅ 完成 |
| Sintel | `data/sintel/` | 直接使用 | ✅ 完成 |
| KITTI | `data/long_kitti_s1/` | ✅ | ✅ 完成 |
| Bonn | `data/long_bonn_s1/` | ✅ | ✅ 完成 |
| 7scenes | 已下载 | ✅ (18 seqs, 7 scenes) | ✅ 完成 |

**运行命令**：
```bash
conda activate ttt3r
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src accelerate launch --num_processes 1 --main_process_port 29560 \
    eval/relpose/launch.py \
    --weights model/cut3r_512_dpt_4_64.pth \
    --output_dir eval_results/relpose/<dataset>/<config> \
    --eval_dataset <dataset> --size 512 --model_update_type <config> \
    --ortho_alpha_novel 0.5 --ortho_alpha_drift 0.05 --ortho_beta 0.95
```
