# DDD3R — Experiment Results & Analysis

## Relpose ATE — Long Sequence (1000f)

**TUM（8 sequences）**

| Config | ATE ↓ | vs cut3r | 角色 |
|--------|-------|----------|------|
| cut3r (baseline) | 0.166 | — | |
| ttt3r | 0.103 | -38.1% | existing method |
| constant (p=0.33) | 0.066 | -60.2% | M1 severity evidence |
| brake | 0.063 | -61.8% | robust default |
| **ortho** | **0.056** | **-66.5%** | precision variant |
| ddd3r_entropy | 0.070 | -57.8% | attention entropy adaptive |
| **ddd3r_de** | **0.057** | **-65.5%** | drift energy adaptive |

**ScanNet（96 scenes, 65 valid）**

| Config | ATE ↓ | vs cut3r | 角色 |
|--------|-------|----------|------|
| cut3r (baseline) | 0.817 | — | |
| ttt3r | 0.406 | -50.3% | existing method |
| constant (p=0.5) | 0.280 | -65.8% | M1 severity evidence |
| **brake** | **0.261** | **-68.0%** | robust default |
| ortho | 0.492 | -39.8% | ⚠ 退化（drift energy 高）|
| ddd3r_entropy | 0.294 | -64.0% | attention entropy adaptive |
| ddd3r_de | 0.374 | -54.2% | drift energy adaptive（高 drift 场景退化）|

**Sintel（14 sequences, ~20-50f）**— 序列极短，over-update 尚未累积，任何 dampening 均无益。

**KITTI Odom（11 sequences, 全长 271-4661f, OOD outdoor）**

| Config | ATE Mean ↓ | vs cut3r | ATE RMSE ↓ | vs cut3r |
|--------|-----------|----------|-----------|----------|
| cut3r | 192.84 | — | 212.59 | — |
| ttt3r | 161.14 | -16.4% | 183.16 | -13.8% |
| constant | 156.61 | -18.8% | 179.33 | -15.6% |
| brake | 166.89 | -13.5% | 185.68 | -12.7% |
| ortho (γ=0) | 154.10 | -20.1% | 170.03 | -20.0% |
| **ddd3r_g1** | **149.05** | **-22.7%** | **166.48** | **-21.7%** |
| ddd3r_g3 | 154.18 | -20.0% | 169.52 | -20.3% |
| auto_warmup_linear | 152.33 | -21.0% | 170.00 | -20.0% |

KITTI 官方指标（segment-based）：

| Config | t_err (%) ↓ | r_err (deg/100m) ↓ |
|--------|-------------|---------------------|
| cut3r | 93.94 | 22.66 |
| ttt3r | 95.66 | 21.09 |
| brake | 88.17 | 16.75 |
| **ortho** | **86.77** | **9.38 (-58.6%)** |
| ddd3r_g1 | 91.67 | 18.28 |
| ddd3r_g5 | 86.75 | 12.26 |

关键发现：
- **ddd3r_g1 (γ=1) ATE 全场最优** (-22.7%)，唯一拯救 seq01 highway (449 vs ortho 642)
- **Ortho 在 KITTI 官方指标碾压**：rotation error -58.6%，但 ATE 和 KITTI 官方给出不同 winner
- **Indoor/outdoor 最优方法不同**：indoor brake/constant 最优，outdoor ortho-family 最优 → 支撑自适应方法必要性
- **γ=1 是 KITTI sweet spot**（与 indoor 偏好不同），γ=3 variance 最低 (std=69.63)
- Seq01 (highway) 所有方法崩盘，brake/ortho 甚至 regress vs cut3r
- 完整 14 methods × 11 sequences 数据见 `eval/relpose/kitti_odo_full_report.md`（来源：zjc 分支）

## Relpose ATE — Short Sequence (90f)

**TUM 90f** ✅ 已验证 (2025-04-03, 8 scenes)

| Config | ATE ↓ | vs cut3r | vs ttt3r |
|--------|-------|----------|----------|
| cut3r | 0.0325 | — | — |
| ttt3r | 0.0189 | -41.8% | — |
| constant | 0.0160 | -50.8% | -15.3% |
| brake | 0.0153 | -52.8% | -19.0% |
| **ortho (γ=0)** | **0.0145** | **-55.4%** | **-23.3%** |
| entropy | 0.0157 | -51.8% | -16.9% |

TTT3R 论文报告 TUM 90f CUT3R=0.046, TTT3R=0.028，与本表差异 ~40%。重新运行两次结果一致（0.0325/0.0189），差异来自 TUM 数据预处理/GT 格式不同。

TUM 90f 关键观察：
- **与 ScanNet 90f 相反**：TUM 90f DDD3R 方法全部优于 ttt3r，ortho 最优 (-23.3%)
- TUM local drift energy 低 (0.40)：相邻帧方向多样化，ortho 分解有效抑制真正的 drift → 即使短序列也有增益
- ScanNet local drift energy 高 (0.60)：相邻帧方向高度一致（有用 refinement），ortho 误抑制 → 短序列反而退化
- 注意：此处 drift energy 指 local cos(δ_t, δ_{t-1})²，非 EMA drift energy（见 A4c）

**ScanNet 90f** ✅ 已验证 (2025-04-03, 90 scenes)

| Config | ATE ↓ | vs cut3r | vs ttt3r |
|--------|-------|----------|----------|
| cut3r | 0.0948 | — | — |
| ttt3r | 0.0641 | -32.4% | — |
| constant | 0.0645 | -31.9% | +0.7% |
| brake | 0.0721 | -24.0% | +12.5% |
| ortho (γ=0) | 0.0870 | -8.2% | +35.8% |
| entropy | 0.0683 | -28.0% | +6.5% |

评测 pipeline 与 TTT3R 论文一致（scannet_s3_90, data/long_scannet_s3, ScanNet test 100 scenes → 90 valid）。TTT3R 论文报告 CUT3R=0.099, TTT3R=0.064；本表 CUT3R 略低 (~4%，可能帧采样微小差异），TTT3R 完全一致。

90f 关键观察：
- **短序列 DDD3R 无优势**：constant ≈ ttt3r，brake/ortho/entropy 均劣于 ttt3r
- **Ortho 在 90f 表现最差** (-8.2% vs cut3r)，方向分解在无显著 drift 时反而有害
- 与 1000f 结论完全相反（1000f: constant -66%, brake -68%, ortho -39%）→ 支撑 "DDD3R 随长度增益递增" 的核心叙事

## ScanNet Scaling Curve — ATE vs Sequence Length

| Method | 200f (n=67) | 500f (n=65) | 1000f (n=65) | 200→1000 倍数 |
|--------|-------------|-------------|--------------|--------------|
| cut3r | 0.320 | 0.667 | 0.817 | 2.6x |
| ttt3r | 0.143 | 0.277 | 0.406 | 2.8x |
| **constant** | **0.138** | **0.214** | 0.280 | 2.0x |
| brake | 0.176 | 0.282 | **0.261** | 1.5x |
| ortho (γ=0) | 0.240 | 0.409 | 0.492 | 2.1x |
| ddd3r_entropy | 0.154 | 0.242 | 0.294 | 1.9x |

Scaling curve 关键观察：
- Over-update 随长度加剧，所有方法 ATE 均随长度增长
- Constant 在 200f/500f 一致最优；brake 在 1000f 反超（非单调：500f 退化后 1000f 回升）
- Pure ortho 持续恶化（drift 累积），entropy 介于 constant 和 ttt3r 之间

**注意**：brake 的 200f/500f 数据用 τ=2.0 (ddd3r_brake)，1000f 用 τ=1.0 (ttt3r_momentum_inv_t1)。τ=1 结果更优。

## Video Depth — Abs Rel ↓ (scale&shift alignment)

| Config | KITTI | Bonn | Sintel |
|--------|-------|------|--------|
| cut3r | 0.1190 | 0.0819 | 0.4674 |
| ttt3r | 0.1074 (-9.7%) | 0.0720 (-12.1%) | 0.4332 (-7.3%) |
| brake | 0.1061 (-10.8%) | 0.0658 (-19.6%) | 0.4022 (-13.9%) |
| ortho | 0.1042 (-12.4%) | 0.0680 (-17.0%) | 0.4175 (-10.7%) |
| constant | — | — | 0.4078 (-12.7%) |

## 3D Reconstruction — 7scenes

| Config | Acc ↓ | Comp ↓ | NC ↑ |
|--------|-------|--------|------|
| cut3r | 0.092 | 0.048 | 0.563 |
| ttt3r | 0.027 (-70.7%) | 0.023 (-52.1%) | 0.581 |
| brake | 0.021 (-77.2%) | 0.022 (-54.2%) | 0.580 |
| ortho | 0.026 (-71.7%) | 0.022 (-54.2%) | 0.577 |

## Inference Overhead — TUM (3 seqs × 200f, 3 repeats)

| Config | FPS | Peak Mem (GB) | vs cut3r |
|--------|-----|---------------|----------|
| cut3r | 8.44 | 6.14 | — |
| ttt3r | 9.82 | 6.14 | +16% faster |
| brake | 10.03 | 6.14 | +19% faster |
| ortho | 9.95 | 6.14 | +18% faster |

所有方法**零额外内存**，速度甚至略快。

---

## Analysis

### A1-A3: Scalar Gate 退化为常数（支撑 M2）
- **A1**: TTT3R 所有 scalar adaptive gate 退化为 ~constant 0.33（σ≈0.02），无时序动态
- **A2**: cos variance vs improvement 无相关（r=-0.13, p=0.63）→ adaptive timing 无意义
- **A3**: Scalar gate per-scene ~50/50 改善/退化 → 不 robust

### A4: Delta Direction — ScanNet vs TUM（支撑 M3）

**A4a: 相邻帧方向一致性** (cos(δ_t, δ_{t-1}), ttt3r model, 全场景均值)

| 指标 | TUM (8 scenes) | ScanNet (96 scenes) |
|------|----------------|---------------------|
| cos(δ_t, δ_{t-1}) mean | 0.617 ± 0.037 | 0.767 ± 0.037 |
| local drift energy (cos²) | 0.398 ± 0.041 | 0.598 ± 0.054 |

ScanNet 相邻帧方向更一致（0.77 vs 0.62）。

**A4c: 实际自适应信号** (cos(δ_dir, ema_drift_dir)² 的 EMA, ddd3r model, β=0.95)

⚠️ **与 A4a 的 "drift energy" 是不同量**。A4a 是相邻帧 cos²（局部），A4c 是与 EMA drift 方向的 cos² 再做 EMA（累积）。

| 指标 | TUM 90f | ScanNet 90f | TUM 1000f | ScanNet 1000f |
|------|---------|-------------|-----------|---------------|
| EMA drift_e mean | **0.786** | **0.660** | **0.471** | **0.506** |
| drift_e std | 0.062 | 0.108 | 0.158 | 0.131 |
| frac(de>0.5) | 0.90 | 0.81 | 0.42 | 0.52 |
| frac(de>0.7) | 0.76 | 0.47 | 0.19 | 0.17 |

**关键发现**：
1. **TUM EMA drift_e 反而更高** (0.786 vs 0.660 on 90f) — 与 A4a 的方向相反！
2. **Drift energy 随序列长度单调递减**（TUM 0.79→0.47, ScanNet 0.66→0.51），非恒定
3. **TUM/ScanNet 分布高度重叠**（5-95th pct 重叠区覆盖 69% TUM、89% ScanNet frames）
4. Per-token std ~0.16 两个数据集几乎一致 → token 级别也无区分力

**为什么两种指标方向相反**：
- A4a `cos(δ_t, δ_{t-1})`：ScanNet > TUM → ScanNet 相邻帧方向更一致
- A4c `cos(δ, ema_drift_dir)²`：TUM > ScanNet → TUM 与累积方向更对齐
- 原因：TUM 场景运动简单（旋转主导），EMA drift direction 准确追踪运动方向 → 高对齐；ScanNet 场景结构复杂，即使相邻帧一致，但 EMA 方向偏移后每帧与 EMA 的对齐度反而降低

**⚠️ 对自适应方法的影响**：
- DDD3R 实际代码中的自适应信号 (A4c) 无法区分 TUM 和 ScanNet → 解释了所有 drift-energy-based 自适应方法失败的根因
- A4a 的 local drift energy 有区分力（TUM=0.40 vs ScanNet=0.60），但代码没用这个信号
- 可能需要切换到 local cos(δ_t, δ_{t-1}) 作为自适应信号，或放弃 per-frame 自适应

### A4b: Local Drift Energy Scaling — 随序列长度恒定

| Dataset | 200f | 500f | 1000f | 趋势 |
|---------|------|------|-------|------|
| TUM cos(δ_t, δ_{t-1}) | 0.618 ± 0.037 | 0.616 ± 0.048 | 0.626 ± 0.049 | ≈ 恒定 |
| TUM local drift_e | 0.398 ± 0.041 | 0.397 ± 0.055 | 0.409 ± 0.057 | ≈ 恒定 |
| ScanNet cos(δ_t, δ_{t-1}) | 0.767 ± 0.037 | 0.774 ± 0.031 | 0.775 ± 0.030 | ≈ 恒定 |
| ScanNet local drift_e | 0.598 ± 0.055 | 0.607 ± 0.046 | 0.609 ± 0.043 | ≈ 恒定 |

Local drift energy 确实是场景固有属性，不随长度变化。Over-update 加剧源于恒定量级误差的逐帧累积。

**对方法设计的启示**：
1. Dampening magnitude（constant α）直接减缓累积速度，是 over-update 的充分解
2. 方向分解在 TUM（local drift_e 低=0.40）有效，在 ScanNet（高=0.60）有害
3. 实际代码中的 EMA drift energy 不是有效的自适应信号 — 需要改用 local drift energy 或其他信号

### A5: TTSA3R TAUM Gate 也退化为常数（支撑 M2）
TAUM temporal gate σ_time=0.006（比 ttt3r 的 σ≈0.02 小 3-4x），更严重退化为常数 ~0.355。
理论原因：`state_change / mean(state_change)` 归一化后均值恒为 1.0，sigmoid(1-1.5) ≈ 0.378。

### A6: Over-update 普遍存在（支撑 M1）
| | TUM 90f | TUM 1000f | ScanNet 90f | ScanNet 1000f | Sintel ~50f | KITTI full |
|---|---------|-----------|-------------|---------------|-------------|-----------|
| cut3r ATE | 0.033 | 0.166 | 0.095 | 0.805 | 0.209 | 192.84 |
| ttt3r vs cut3r | -42% | -38% | -32% | -50% | 0% | -16.4% |
| constant vs cut3r | -53% | -60% | -32% | -66% | +5% | -18.8% |
| brake vs cut3r | -53% | -62% | -24% | -68% | +14% | -13.5% |
| ortho vs cut3r | -55% | -66% | -8% | -40% | +13% | -20.1% |
| ddd3r_g1 vs cut3r | — | — | — | — | — | **-22.7%** |

Over-update 随长度加剧：ScanNet 1000f/90f = 8.5x，TUM = 5.0x。Sintel 极短无 over-update。KITTI outdoor OOD 同样 over-update 严重，且 ortho-family 在 outdoor 反超 brake（与 indoor 相反）。

### A7: Per-Scene Scatter — Drift Energy vs Improvement（ScanNet 90f, 90 valid）
| 指标 | Ortho | Brake | Constant |
|------|-------|-------|----------|
| 改善 scene 数 | 58/90 (64%) | 72/90 (80%) | 77/90 (86%) |
| r (drift energy vs improvement) | +0.248 (p=0.018) | +0.157 (p=0.14) | +0.091 (p=0.39) |

Ortho 与 drift energy 显著正相关：drift energy 高 → 退化。Brake/constant 对 drift energy 不敏感。

### A8: Cross-Dataset Method Ranking 反转（支撑自适应方法）

各方法在不同数据集的排名（ATE，1=最优）：

| Method | TUM 1000f | ScanNet 1000f | KITTI full | 排名模式 |
|--------|-----------|---------------|------------|----------|
| constant | 3 | 2 | 3 | 稳定中游 |
| brake | 2 | **1** | 5 | indoor 最优，outdoor 最差 |
| ortho (γ=0) | **1** | 5 | 2 | 低 drift 最优，高 drift 退化 |
| ddd3r_g1 | — | 4 | **1** | KITTI 专属最优 |

**关键观察**：
- **没有任何 fixed 方法跨数据集一致最优**。Brake 在 indoor (ScanNet) 第一但 KITTI 第五；ortho 在 TUM 第一但 ScanNet 第五。
- **排名反转根源是 local drift energy 差异**：TUM local_de≈0.40（低 → ortho 有效），ScanNet local_de≈0.60（高 → ortho 误抑制），KITTI outdoor 动态更复杂。注意：DDD3R 代码中的 EMA drift energy 无法区分两个数据集（见 A4c），local drift energy 有区分力但代码未使用。
- **这是 DDD3R 自适应方法的核心论据**：任何 fixed 配置都无法跨场景鲁棒，需要根据在线 drift energy 动态调节。

### A9: KITTI ATE vs 官方指标分离（local vs global tradeoff）

| 指标类型 | 最优方法 | 核心含义 |
|----------|----------|----------|
| ATE Mean (global) | ddd3r_g1 (149.05) | 全局轨迹一致性 |
| ATE RMSE (global) | ddd3r_g1 (166.48) | 全局轨迹一致性（outlier 敏感） |
| KITTI t_err (segment) | ortho / ddd3r_g5 (86.75%) | 局部平移精度 |
| KITTI r_err (segment) | ortho (9.38 deg/100m) | 局部旋转精度 |
| RPE (frame-to-frame) | cut3r (2.09m) | 逐帧精度 |

**分析**：
- Ortho-family 的方向分解显著改善**旋转估计** (r_err -58.6%)，因为 drift 方向主要对应累积旋转误差，投影抑制直接消除旋转 drift。
- ATE 和 KITTI 官方指标给出不同 winner：KITTI 官方是 segment-based (100-800m)，对全局漂移更宽容；ATE 是全局 Sim(3) 对齐后的绝对误差。
- RPE 最优仍是 cut3r：dampening 牺牲了逐帧精度换取全局一致性，这是 **local-global tradeoff** 的固有特征。
- **实际应用意义**：SLAM/导航关注 ATE（全局一致性），motion planning 关注 RPE/KITTI 指标（局部精度）。DDD3R 允许通过 γ 调节 tradeoff 位置。

### A10: KITTI Seq01 Highway — 所有方法的共同瓶颈

| Method | Seq01 ATE | vs cut3r | Seq01 以外 Mean |
|--------|-----------|----------|-----------------|
| cut3r | 600.32 | — | 152.09 |
| brake | 643.76 | **+7.2%** | 119.20 |
| ortho | 642.26 | **+7.0%** | 105.29 |
| ddd3r_g1 | **449.58** | **-25.1%** | 119.01 |
| auto_warmup_linear | 466.98 | -22.2% | 120.87 |

**分析**：
- Highway 场景 (seq01, 1101f) 是极端 outlier：高速直行、视觉特征匮乏、camera 几乎纯平移。
- Brake 和 ortho 在 seq01 **反而 regress**（比 cut3r 更差 7%）：brake 的 cos 信号在快速变化的 update 方向下不稳定；ortho 的 EMA drift direction 在特征稀疏时估计不准。
- γ=1 唯一显著缓解 seq01 (-25.1%)：适度的 steep 衰减让高 drift-energy token 也保留部分信号，避免过度抑制。
- **排除 seq01 后**，ortho 以 105.29 大幅领先所有方法，说明 ortho 的核心机制在正常 driving 场景高度有效。
- **启示**：Highway 是 model-level 的 OOD 问题（indoor 训练 → outdoor featureless），非 update rule 能解决的。Paper 应明确此 limitation。

### A11: γ Sweep — 最优 γ 与 drift energy 相关性

| Dataset | local drift_e | 最优 γ | 解释 |
|---------|---------------|--------|------|
| TUM | 0.40 | ≥3 (0.054) | 低 local drift → 激进 ortho 有益 |
| ScanNet | 0.60 | 0 或 constant/brake | 高 local drift → 方向分解有害 |
| KITTI | ？(outdoor) | 1 (149.05) | 中等 → 适度分解 |

**趋势**：local drift energy 越低，最优 γ 越大（越偏 ortho）；越高，最优 γ 越小（越偏 isotropic）。
- 低 local drift energy = 相邻帧方向多样化 = 可以放心抑制 drift 分量
- 高 local drift energy = 相邻帧方向一致（有用 refinement）= 抑制 drift 会丢信息

**⚠️ 但 ddd3r_de 自适应失败**：代码使用的 EMA drift energy 在 TUM/ScanNet 上分布重叠（A4c），无法区分。Local drift energy 有区分力但代码未使用。

### A12: 方法设计 hierarchy — Dampening 三阶段的增量贡献

| 阶段 | 代表方法 | TUM 1000f | ScanNet 1000f | KITTI full | 增量贡献 |
|------|----------|-----------|---------------|------------|----------|
| 0. No dampening | cut3r | 0.166 | 0.817 | 192.84 | — |
| 1. Scalar dampening | ttt3r | 0.103 (-38%) | 0.406 (-50%) | 161.14 (-16%) | 方向无关的 update 缩放 |
| 2. Constant magnitude | constant | 0.066 (-60%) | 0.280 (-66%) | 156.61 (-19%) | 去掉无效 adaptive gate |
| 3a. 方向无关 adaptive | brake | 0.063 (-62%) | **0.261 (-68%)** | 166.89 (-13%) | cos-based stability 信号 |
| 3b. 方向分解 fixed | ortho (γ=0) | **0.056 (-66%)** | 0.492 (-40%) | 154.10 (-20%) | drift/novel 分解 |
| 3c. 方向分解 + steep | ddd3r_g1 | 0.056 (-66%) | 0.358 (-56%) | **149.05 (-23%)** | γ 调节 ortho 强度 |
| 4. 自适应 | ddd3r_de | 0.057 (-66%) | 0.374 (-54%) | — | drift energy 驱动（ScanNet 退化） |

**关键 takeaway**：
1. **Stage 1→2 提升最大**（constant 甚至优于 ttt3r adaptive）：证明 over-update severity (M1) 是主要问题，现有 adaptive gate 无用 (M2)。
2. **Stage 2→3 分化**：brake vs ortho 在不同数据集反转，证明方向性质 (M3) 决定了最优策略。
3. **Stage 3→4 是 DDD3R 的核心价值**：统一框架下，通过 drift energy 自动选择 brake-like (高 drift_e) 或 ortho-like (低 drift_e) 行为。

### Qualitative
- **Depth** (Bonn balloon2): CUT3R 后期退化 (0.089→0.107)，brake/ortho 保持稳定 (~0.04)
- **Trajectory** (ScanNet 1000f): CUT3R 轨迹大幅漂移，brake/ortho 紧贴 GT（ATE 降低 85-90%）
- **Trajectory** (KITTI seq03): cut3r ATE 163.6 → ortho 24.8 (-84.8%)，最大单序列改善

### Ortho Hyperparameter Sensitivity
- **TUM**: α_drift 鲁棒 (0.05-0.2 <2%)，α_novel≥0.5 鲁棒，β=0.95 sweet spot
- **ScanNet**: 与 TUM 完全反转 — β=0.99 最优，α_drift 越高越好。证实 drift 性质根本不同。

---

## Steep Adaptive（e^γ 公式，完整结果）

**公式**：`w = ē^γ`（保守）。γ→∞ = pure ortho, γ→0 = isotropic。

**完整 Spectrum 表格（所有 γ 值）：**

| DDD3R Config | TUM 1000f | ScanNet 1000f | KITTI full (11 seqs, ATE Mean) |
|---|---|---|---|
| cut3r (baseline) | 0.166 | 0.817 | 192.84 |
| ttt3r | 0.103 | 0.406 | 161.14 |
| constant (α⊥=α∥) | 0.066 | **0.280** | 156.61 |
| brake | 0.063 | **0.261** | 166.89 |
| γ=0 (pure ortho) | 0.056 | 0.492 | 154.10 |
| γ=1 | 0.056 | 0.358 | **149.05** |
| γ=2 | 0.056 | 0.394 | 155.19 |
| γ=3 | **0.054** | 0.407 | 154.18 |
| γ=5 | 0.055 | 0.456 | 159.46 |

**关键发现：**
- **γ=1 是 KITTI 全序列最优** (149.05, -22.7%)，但 ScanNet 修复不够 (0.358 vs brake 0.261)。
- **没有单一 γ 在所有数据集都最优**。TUM 偏好大 γ (≥3)，ScanNet 偏好小 γ 或 constant/brake，KITTI 偏好 γ=1。
- **Indoor vs outdoor 最优 γ 完全不同**：进一步证明 drift energy 性质差异 → 自适应方法必要。
- **Steep self-correction 在 ScanNet 上不够激进** — 即使低 γ 也无法完全退化为 isotropic。
- **Constant/brake 仍是 indoor robust default**：ScanNet 上最优或接近最优。

### Constant Dampening per-dataset 最优 α
| 数据集 | 最优 α |
|--------|--------|
| TUM | 0.33 |
| ScanNet | 0.5 |

---

## Adaptive Methods Comparison（自适应方法对比）

### Attention Entropy Adaptive (auto_gamma=entropy)

**方案**：用 cross-attention 归一化 entropy h̄_t 插值 α_∥^(t) = h̄_t·α_⊥ + (1-h̄_t)·α_∥。

| Config | TUM 1000f | ScanNet 1000f |
|--------|-----------|---------------|
| pure ortho (γ=0) | 0.056 | 0.492 |
| brake | 0.063 | **0.261** |
| constant | 0.066 | 0.280 |
| **ddd3r_entropy** | 0.070 | 0.294 |
| ddd3r_entropy β=0.9 | 0.068 | 0.295 |
| ddd3r_entropy β=0.99 | — | 0.295 |

**结论**：Entropy 在 ScanNet 上有效（0.492→0.294），但在 TUM 上比 constant 还差（0.070 vs 0.066）。Attention entropy 作为 drift energy 的 proxy 不够准确。

### Drift Energy Adaptive (auto_gamma=drift_energy) — 新方案

**方案**：直接用在线 drift energy e_t = EMA(cos²(δ_t, d_t)) 插值 α_∥^(t) = e_t·α_⊥ + (1-e_t)·α_∥。零额外成本（EMA drift direction 和 cos² 已经在 decompose 步骤中计算）。

| Config | TUM 1000f | ScanNet 1000f |
|--------|-----------|---------------|
| pure ortho (γ=0) | 0.056 | 0.492 |
| brake | 0.063 | **0.261** |
| constant | 0.066 | 0.280 |
| ddd3r_entropy | 0.070 | 0.294 |
| **ddd3r_de** | **0.057** | 0.374 |

**TUM 结果**：0.057，接近 pure ortho (0.056)，大幅优于 entropy (0.070)。TUM EMA drift_e≈0.47 (1000f) → α_∥_eff 偏 ortho。

**ScanNet 结果**：0.374，显著差于 brake (0.261) 和 constant (0.280)。ddd3r_de 仅赢 ortho (53/65 scenes)，输 constant (50/65) 和 brake (61/65)。

**自适应失败的根因** (A4c 发现)：EMA drift energy 在 TUM 和 ScanNet 上高度重叠（TUM 0.47 vs ScanNet 0.51 at 1000f），无法区分两个数据集 → ddd3r_de 在两个数据集上看到类似的信号，无法做出差异化决策。drift_e 与退化程度无相关（r=-0.08），进一步确认 EMA drift energy 不是有效的自适应信号。

### Auto-gamma 其他变体（完整结果，含 KITTI full）

| Config | TUM 1000f | ScanNet 1000f | KITTI full (ATE Mean) |
|--------|-----------|---------------|----------------------|
| warmup_linear | 0.056 | 0.360 | **152.33 (-21.0%)** |
| warmup_threshold | 0.079 | **0.270** | 159.09 |
| steep_sigmoid | 0.055 | 0.342 | 157.54 |
| steep_sigmoid_k20 | **0.054** | 0.380 | — |
| steep_clamp | 0.055 | 0.361 | 168.88 |

Warmup_linear 是 KITTI 上最优 auto-gamma (152.33, 仅次于 fixed ddd3r_g1)。Warmup_threshold 在 ScanNet 最优但 TUM/KITTI 退化。都没有跨数据集完美平衡。

### Local Drift Energy Adaptive (auto_gamma=local_de) — 新信号探索

**动机**：A4c 发现 EMA drift energy 无法区分 TUM/ScanNet（高度重叠），而 A4a 的 local drift energy `cos(δ_t, δ_{t-1})²` 有稳定 gap（TUM≈0.40 vs ScanNet≈0.60，跨长度恒定）。因此尝试用 local DE 替代 EMA DE 作为自适应信号。

**实现**：per-frame 计算 `cos(δ_t, δ_{t-1})²`，EMA 平滑后插值 α_∥：
- high local_de (ScanNet-like) → α_∥ → α_⊥ (isotropic, less suppression)
- low local_de (TUM-like) → α_∥ stays small (ortho suppression)

| Config | TUM 90f | ScanNet 90f | TUM 1000f | ScanNet 1000f |
|--------|---------|-------------|-----------|---------------|
| pure ortho (γ=0) | **0.015** | 0.087 | **0.056** | 0.492 |
| brake | 0.015 | 0.072 | 0.063 | **0.261** |
| constant | 0.016 | 0.065 | 0.066 | 0.280 |
| ddd3r_entropy | 0.016 | 0.068 | 0.070 | 0.294 |
| ddd3r_de (EMA) | — | — | 0.057 | 0.374 |
| **local_de (linear)** | 0.016 | 0.069 | 0.067 | 0.297 |
| **local_de_sig (k=20)** | 0.016 | **0.065** | 0.075 | — |

**分析**：
1. **ScanNet 1000f**: local_de (0.297) 比 ddd3r_de (0.374) 改善 20%，接近 entropy (0.294)，但仍不如 constant (0.280) 和 brake (0.261)。
2. **TUM 1000f**: local_de (0.067) ≈ constant (0.066)，远不如 ortho (0.056) 和 ddd3r_de (0.057)。EMA local_de 收敛后约 0.4-0.5，导致 α_∥_eff ≈ 0.23，远高于 ortho 的 0.05。
3. **Sigmoid variant**: ScanNet 90f 上 0.065（=constant），但 TUM 1000f 退化到 0.075——sigmoid sharpening 在短序列有效但长序列过于 aggressive。

**自适应天花板分析**：
- 所有自适应方法在 α_∥∈[0.05, 0.50] 之间插值
- TUM oracle = ortho (0.056)，自适应最多 match 它
- ScanNet oracle = brake (0.261)，自适应需要 fully isotropic 才能接近，但 local_de 收敛慢 + EMA early frame bias 导致信号不够 aggressive
- **根本限制**：场景级别的 scalar 信号无法替代 brake 的 per-token spatial gating

**结论**：Local DE 信号方向正确（比 EMA DE 好 20%），但无法实现跨数据集突破。自适应的核心价值在于证明排名反转现象和分析信号差异，而非实现数值上超越 fixed configs。Per-token 级别的自适应（如 brake 的空间 gating）是更有前景的方向（future work）。
