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

**ScanNet（96 scenes, 65 valid）**

| Config | ATE ↓ | vs cut3r | 角色 |
|--------|-------|----------|------|
| cut3r (baseline) | 0.817 | — | |
| ttt3r | 0.406 | -50.3% | existing method |
| constant (p=0.5) | 0.280 | -65.8% | M1 severity evidence |
| **brake** | **0.261** | **-68.0%** | robust default |
| ortho | 0.492 | -39.8% | ⚠ 退化（drift energy 高）|

**Sintel（14 sequences, ~20-50f）**— 序列极短，over-update 尚未累积，任何 dampening 均无益。

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

## Video Depth — Abs Rel ↓

| Config | KITTI | Bonn | Sintel |
|--------|-------|------|--------|
| cut3r | 0.1515 | 0.0990 | 1.0217 |
| ttt3r | 0.1319 (-12.9%) | 0.0997 | 0.9776 (-4.3%) |
| brake | 0.1061 (-30.0%) | 0.0658 (-33.5%) | 0.4022 (-60.6%) |
| ortho | 0.1042 (-31.2%) | 0.0680 (-31.3%) | 0.4175 (-59.1%) |

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

**Spectrum 展示表格（experiments section, e^γ 公式）：**

| DDD3R ($\gamma$) | TUM 1000f | ScanNet 1000f | KITTI seq04 | KITTI seq03 | 行为 |
|---|---|---|---|---|---|
| α⊥=α∥=0.5 (constant) | 0.066 | **0.280** | **5.10** | 60.68 | 无方向感知 |
| brake (baseline) | 0.063 | **0.261** | 7.05 | **25.38** | 纯时序自适应 |
| γ=0.5 | 0.061 | 0.327 | **4.89** | 46.54 | 轻度 ortho |
| γ=1 | 0.056 | 0.358 | 6.92 | 37.89 | 适度 ortho |
| γ=2 | 0.056 | 0.394 | 8.78 | **24.58** | 较强 ortho |
| γ=3 | **0.054** | 0.407 | 8.14 | 26.79 | 强 ortho |
| γ=5 | 0.055 | 0.456 | 10.37 | 26.01 | 接近 pure ortho |
| γ=0 (pure ortho) | 0.056 | 0.492 | 17.79 | 39.91 | 全方向分解 |

**γ 方向性**：γ→∞ = pure ortho, γ→0 = isotropic。TUM (e≈0.4) 偏好大 γ (≥2)，ScanNet (e≈0.6) 偏好小 γ 或 constant/brake。没有单一 γ 跨数据集最优，steep self-correction 力度不足。

### Constant Dampening per-dataset 最优 α
| 数据集 | 最优 α |
|--------|--------|
| TUM | 0.33 |
| ScanNet | 0.5 |
