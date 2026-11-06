# DDD3R — Paper Narrative & Route Decisions

## Paper Narrative

### 大问题（一句话）
**Recurrent 3D reconstruction 的 state update 缺乏有效调节，导致系统性退化。**

### 三个 Manifestation（递进关系）

**M1: 幅度失控。** Update 整体过大，90f 即可观察，随序列长度持续累积（ScanNet 8.5x, TUM 5.0x）。常数 dampening 即可改善 30-68%，证明 over-update 是 dominant failure mode。

**M2: 现有自适应 gate 全部失效。** TTT3R sigmoid gate σ≈0.02, TTSA3R TAUM σ≈0.006，均退化为近似常数。理论原因明确（sigmoid 饱和 / 归一化后均值恒为1）。这解释了为什么问题没被已有工作解决。

**M3: Update 方向包含结构性冗余。** Delta 中大量沿历史方向的 drift 分量，但性质因场景而异（TUM drift energy ~40% = 有害重复, ScanNet ~60% = 有用 refinement）。光控制幅度不够，还需要理解方向。

递进：M1"有多严重" → M2"为什么没被解决" → M3"更深层是什么"。

### 统一框架：DDD3R
一个 training-free、plug-in、zero-overhead 的 state update 调节框架。核心定位：**理解和解决 state update 调节困境的统一框架**，而非单一"最优算法"。

**统一公式：**
```
S_t = S_{t-1} + β_t (α⊥ · δ⊥ + α∥ · δ∥)
```
- α⊥ = α∥ → constant dampening（无方向感知，回应 M1）
- α⊥ > α∥, γ=0 → fixed directional decomposition（回应 M1+M3，low-drift 最优）
- α⊥ > α∥, γ>0 → drift-energy-modulated（spectrum 探索工具，非推荐配置）
- Temporal brake → directional decomposition 的一阶近似（robust default，跨数据集最稳定）

**Practitioner guide:**
- 未知场景 → temporal brake（safe default，全 benchmark 1st-2nd）
- 已知 low-drift（如慢速 indoor scanning）→ full decomposition γ=0（额外 10-15% 提升）
- Outdoor driving → γ=1（KITTI sweet spot）

### Contributions（三个）
**C1 诊断:** 揭示 recurrent 3D reconstruction 中 state update 调节的三重失效——幅度失控、自适应 gate 退化为常数、方向冗余——并证明 over-update 是 dominant failure mode。

**C2 统一框架:** 提出 DDD3R，training-free、zero-overhead 的统一调节框架。一个 update rule 定义 continuous spectrum，从 isotropic dampening 到 full directional decomposition，统摄所有方法变体为 special cases。

**C3 实验发现:** 发现 cross-dataset ranking reversal 现象——最优 operating point 随数据集系统性反转，drift energy（scene-intrinsic property）governs 这一反转。Temporal brake 是跨 drift regime 的 robust default；directional decomposition 在 low-drift 场景进一步提升 10-15%。

### Paper Structure
- **Section 1 Introduction**: 一个问题，一个框架，三个贡献
- **Section 3 Analysis**: 统一诊断叙事，M1→M2→M3 递进
- **Section 4 Method**: DDD3R 统一 update rule，三个 stage（decompose→reweight→gate）分别回应诊断发现，然后展示 continuous spectrum。γ>0 作为 spectrum 探索工具（非推荐配置）
- **Section 5 Experiments**: DDD3R 整体效果 + spectrum ablation + practitioner guide + ranking reversal 作为 key finding

---

## 路线决策：已确定 — 统一 Spectrum

### 最终路线：Continuous Spectrum（非叠加，非独立）

不走路线 A（叠加式模块）也不走路线 B（独立 operating points），而是**统一 spectrum**：所有方法都是同一个 update rule 的特殊情况，由参数控制在 spectrum 上的位置。

**核心叙事：** DDD3R 不是三个并列方法，而是一个参数化家族。Constant dampening 不是"框架外的 baseline"，而是 α⊥ = α∥ 时方向分解自动消失的退化。Brake 不是并列方法，而是 directional decomposition 的一阶近似。γ>0 作为 spectrum 探索工具验证 continuous structure。

**2026-04-09 叙事调整（NeurIPS 定位）：**

核心转向：从 "we solved it" → "we understood it and provide a principled toolkit"

1. **C3 claim 降级**："drift energy reliably predicts" → "drift energy governs/explains"（诊断性，非预测性）
2. **γ>0 降级**：从 "self-correction mechanism" → "spectrum exploration tool"（验证 spectrum structure，非推荐配置）
3. **Brake 提升**：从 "baseline / 一阶近似" → "robust default across diverse drift regimes"
4. **Practitioner guide 新增**：Experiments 末尾明确给出选择指南
5. **Limitation 加强**：诚实说明所有 online adaptive 方案失败，online selection 是 open problem
6. **Ranking reversal 升格**：从 experiment observation → central empirical finding

**依据**：NeurIPS 偏好 understanding paper > method paper。诊断 (M1-M3) 是真正的核心贡献，spectrum 提供统一语言，ranking reversal 是 key empirical finding。

### Method Section 写作决策记录
- **结构**: Preliminaries → 先给 boxed 总公式 → 三个 Stage 展开 → Continuous Spectrum → Implementation with Algorithm
- **符号**: α∥ (替代 α_drift), α⊥, n×c, Δ_t (大写 matrix), δ_t (per-token), d_t (EMA 后 normalize, projection 对 unit d̂_t)
- **Constant 定位**: α⊥=α∥ 的 exact special case，不在 Table 1 中单独出现
- **Brake 定位**: directional decomposition 的一阶近似，作为 baseline 展示，不作为 DDD3R 组件
- **Steep**: 在 Spectrum subsection 的 paragraph 中介绍，γ 控制 ortho↔constant 连续滑动
- **默认配置**: (α⊥, α∥, β_ema, γ) = (0.5, 0.05, 0.95, 0)，γ>0 作为 ablation
- **Steep 公式**: `w = e^γ`（保守，已对齐代码）。γ→0 isotropic, γ→∞ pure ortho。实验证明没有单一最优 γ，steep 作为 spectrum 展示而非推荐配置
- **Auto-gamma**: 两类方案（warmup sequence-level / steep 公式修复），解决 cross-dataset γ 选择问题。实验进行中
- **Attention Entropy Adaptive (auto_gamma=entropy)**: 第三类方案，用 cross-attention 的归一化 entropy 直接插值 α_∥。h̄_t = EMA(H(attn)/log(N_k))，α_∥^(t) = h̄_t·α_⊥ + (1-h̄_t)·α_∥。entropy 高（scene 在变化）→ α_∥→α_⊥（less suppression, isotropic）；entropy 低（已收敛）→ α_∥ 保持小（aggressive decomposition）。zero-cost：cross-attention 每步已经算好，entropy 只需 softmax+log 额外计算。完全去掉 γ 超参。
