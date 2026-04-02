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
| ddd3r_de | ⬜ running | — | drift energy adaptive |

**Sintel（14 sequences, ~20-50f）**— 序列极短，over-update 尚未累积，任何 dampening 均无益。

**KITTI Odom（11 sequences）**

| Config | seq04 | all 11 seqs (mean) |
|--------|-------|-------------------|
| cut3r | 21.85 | — (只跑了 seq04) |
| ttt3r | 11.06 | — |
| constant | 5.10 | — |
| brake | 7.05 | — |
| ortho | 17.79 | — |
| ddd3r_entropy | 4.93 | 172.92 |

KITTI outdoor long sequences 和 indoor 数据 dynamics 完全不同，DDD3R 自适应方法在大部分 sequences 上崩盘（seq01: 495, seq02: 274）。

## Relpose ATE — Short Sequence (90f)

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

## Video Depth — Abs Rel ↓

| Config | KITTI | Bonn | Sintel |
|--------|-------|------|--------|
| cut3r | 0.1515 | 0.0990 | 1.0217 |
| ttt3r | 0.1319 (-12.9%) | 0.0997 | 0.9776 (-4.3%) |
| brake | 0.1061 (-30.0%) | 0.0658 (-33.5%) | 0.4022 (-60.6%) |
| ortho | 0.1042 (-31.2%) | 0.0680 (-31.3%) | 0.4175 (-59.1%) |
| constant | — | — | 0.4078 (-60.1%) |

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
| 指标 | TUM (8 scenes) | ScanNet (96 scenes) |
|------|----------------|---------------------|
| cos(δ_t, δ_{t-1}) mean | 0.617 ± 0.037 | 0.767 ± 0.037 |
| drift energy (cos²) | 0.398 ± 0.041 | 0.598 ± 0.054 |

ScanNet drift energy 60% vs TUM 40%。ScanNet 的 "drift" 是有用 refinement，ortho 误抑制；TUM 的 drift 是 over-update，ortho 恰好合适。

### A4b: Drift Energy Scaling — 随序列长度恒定（新发现）

| Dataset | 200f | 500f | 1000f | 趋势 |
|---------|------|------|-------|------|
| TUM cos | 0.618 ± 0.037 | 0.616 ± 0.048 | 0.626 ± 0.049 | ≈ 恒定 |
| TUM drift_e | 0.398 ± 0.041 | 0.397 ± 0.055 | 0.409 ± 0.057 | ≈ 恒定 |
| ScanNet cos | 0.767 ± 0.037 | 0.774 ± 0.031 | 0.775 ± 0.030 | ≈ 恒定 |
| ScanNet drift_e | 0.598 ± 0.055 | 0.607 ± 0.046 | 0.609 ± 0.043 | ≈ 恒定 |

**关键发现**：Drift energy 是数据集/场景的固有属性，不随序列长度增长。Over-update 的加剧不是因为 drift 变严重，而是恒定量级误差的逐帧累积（类似随机游走）。

**对方法设计的启示**：
1. Dampening magnitude（constant α）直接减缓累积速度，是 over-update 的充分解
2. 方向分解在低 drift-energy 场景（TUM）额外增益，高 drift-energy 场景（ScanNet）可能有害
3. Drift energy 可在线估计 → 作为自适应信号比 attention entropy 更 principled

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

---

## Steep Adaptive（e^γ 公式，完整结果）

**公式**：`w = ē^γ`（保守）。γ→∞ = pure ortho, γ→0 = isotropic。

**完整 Spectrum 表格（所有 γ 值）：**

| DDD3R Config | TUM 1000f | ScanNet 1000f | KITTI seq04 | KITTI seq03 |
|---|---|---|---|---|
| cut3r (baseline) | 0.166 | 0.817 | 21.85 | 150.29 |
| ttt3r | 0.103 | 0.406 | 11.06 | 103.26 |
| constant (α⊥=α∥) | 0.066 | **0.280** | **5.10** | 60.68 |
| brake | 0.063 | **0.261** | 7.05 | **25.38** |
| γ=0 (pure ortho) | 0.056 | 0.492 | 17.79 | 39.91 |
| γ=0.5 | 0.061 | 0.327 | **4.89** | 46.54 |
| γ=1 | 0.056 | 0.358 | 6.92 | 37.89 |
| γ=2 | 0.056 | 0.394 | 8.78 | **24.58** |
| γ=3 | **0.054** | 0.407 | 8.14 | 26.79 |
| γ=5 | 0.055 | 0.456 | 10.37 | 26.01 |

**关键发现：**
- **γ=1 不是 cross-dataset sweet spot**（预期落空）。TUM OK (0.056)，ScanNet 修复不够 (0.358 vs brake 0.261)。
- **γ=0.5**: KITTI seq04 全场最优 (4.89)，ScanNet 有修复 (0.327 vs ortho 0.492) 但不如 constant/brake。KITTI seq03 退化 (46.54)。
- **没有单一 γ 在所有数据集都最优**。TUM 偏好大 γ (≥2)，ScanNet 偏好小 γ (≤0.5) 或 constant/brake，KITTI 分裂。
- **Steep self-correction 在 ScanNet 上不够激进** — 即使 γ=0.5 (w≈0.77) 也无法完全退化为 isotropic。
- **Constant/brake 仍是 robust default**：constant 和 brake 在 ScanNet/KITTI seq04 上最优或接近最优。

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
| **ddd3r_de** | **0.057** | ⬜ running |

**TUM 结果**：0.057，接近 pure ortho (0.056)，大幅优于 entropy (0.070)。TUM drift_e≈0.40 → α_∥≈0.23，比 entropy 的 α_∥ 更偏 ortho。

**预期 ScanNet**：drift_e≈0.62 → α_∥≈0.33 → 接近 constant (0.280)。如确认，则 ddd3r_de 是首个单一配置跨数据集自动适配的方案。

### Auto-gamma 其他变体（完整结果）

| Config | TUM 1000f | ScanNet 1000f |
|--------|-----------|---------------|
| warmup_linear | 0.056 | 0.360 |
| warmup_threshold | 0.079 | **0.270** |
| steep_sigmoid | 0.055 | 0.342 |
| steep_sigmoid_k20 | **0.054** | 0.380 |
| steep_clamp | 0.055 | 0.361 |

Warmup_threshold 在 ScanNet 最优但 TUM 退化。Steep 系列在 TUM 好但 ScanNet 不够。都没有跨数据集平衡。
