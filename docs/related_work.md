# Related Work — Streaming / Recurrent 3D Reconstruction

整理截止 2026-03-26。按与本项目的关联度排序。

---

## 1. 直接基础（本项目构建于其上）

### CUT3R — Continuous 3D Perception Model with Persistent State
- **来源**: arXiv 2501.12387, 2025-01
- **方法**: Stateful recurrent model，每帧更新 state representation，在线生成 metric-scale pointmaps
- **State 更新**: `state_feat = new_state_feat * 1.0 + old * 0.0`（full replacement）
- **关系**: 我们的 baseline 模型

### TTT3R — 3D Reconstruction as Test-Time Training
- **来源**: arXiv 2509.26645, 2025-09 (revised 2026-03)
- **方法**: 将 CUT3R 的 state update 重新 formulate 为 online learning 问题
- **State 更新**: `ttt3r_mask = sigmoid(mean(cross_attn))`，data-driven per-token gate
- **核心发现**: Alignment confidence 可导出 closed-form learning rate for memory updates
- **局限**: Learned gate 在 image pairs 上训练，未见过 long video 的 temporal redundancy，泛化不足
- **关系**: 我们的直接基础，在其上添加 inference-time gating

---

## 2. 同方向竞品（State/Memory Gating for Recurrent 3D）

### GRS-SLAM3R — Gated Recurrent State SLAM
- **来源**: arXiv 2509.23737, 2025-09
- **方法**: Transformer-based gated update module（update gate + reset gate，类 GRU）
- **Gate 机制**: 两个 transformer gating unit，通过 self-attention + cross-attention 计算 gating weights
- **训练**: 需要训练 gate module
- **结果**: 无 gate → 几何不一致累积；有 gate → 保持空间一致性
- **与我们的区别**: Learned gate vs 我们的 training-free signal-based gate
- **威胁程度**: ⚠️ 高。做的事几乎一样（gated state update），但用 learned approach

### LONG3R — Long Sequence Streaming 3D Reconstruction
- **来源**: arXiv 2507.18255, ICCV 2025
- **方法**: Attention-based memory gating + 3D spatio-temporal memory + dynamic pruning
- **Gate 机制**: Attention-based memory gating 过滤相关 memory → dual-source refined decoder
- **Memory 管理**: 动态裁剪冗余 spatial information，自适应调整分辨率
- **结果**: 减少 27% memory tokens，提速 20%，长序列 SOTA
- **训练**: 需要训练（two-stage curriculum）
- **与我们的区别**: Memory gating（选择哪些 memory 给 decoder）vs 我们的 state update gating（控制 state 更新率）
- **威胁程度**: ⚠️ 中高。Memory gating 方向不同但 narrative 重叠

### OnlineX — Active-to-Stable State Evolution
- **来源**: arXiv 2603.02134, 2026-03（最新）
- **方法**: 将 state 拆分为 active state（高频局部几何）+ stable state（低频全局结构），通过 fusion 传递信息
- **核心 insight**: "state 需要同时处理高频局部几何和低频全局结构，两者有冲突" — 与我们的频域 motivation 高度相似
- **训练**: 需要训练（架构改动）
- **与我们的区别**: 他们用**架构分离**解决高频/低频冲突，我们用**gating**在单一 state 上做调控
- **威胁程度**: ⚠️ 高。Insight 几乎一样，但解法更彻底

---

## 3. Training-Free / Inference-Time 方向

### IncVGGT — Incremental VGGT for Memory-Bounded Long-Range 3D
- **来源**: ICLR 2026
- **方法**: Training-free，top-k relevance-based KV cache pruning，固定 memory budget
- **结果**: 500帧序列上 58.5× fewer operators, 9× lower memory, 4.9× faster vs StreamVGGT
- **可扩展**: 80GB GPU 上可处理 10k 帧
- **与我们的区别**: KV cache pruning（VGGT 系）vs state update gating（CUT3R 系）
- **意义**: 证明 training-free 方向可以发 ICLR 2026 顶会

### XStreamVGGT — Memory-Efficient Streaming Vision Geometry
- **来源**: arXiv 2601.01204, 2026-01
- **方法**: Tuning-free KV cache 压缩（pruning + quantization）
- **结果**: 4.42× memory reduction, 5.48× inference acceleration, negligible performance loss
- **与我们的区别**: KV cache compression vs state update gating

### Test3R — Learning to Reconstruct 3D at Test Time
- **来源**: arXiv 2506.13750, 2025-06
- **方法**: Test-time self-supervised learning，用 image triplet 的几何一致性优化
- **与我们的区别**: Test-time optimization（改权重）vs 我们的 inference-time gating（不改权重）

---

## 4. 其他 Streaming 3D Reconstruction

### Point3R — Explicit Spatial Pointer Memory
- **来源**: arXiv 2507.02863, NeurIPS 2025
- **方法**: Explicit spatial pointer memory，每个 memory slot 有 3D 位置
- **解决**: Implicit memory 的信息丢失问题
- **训练**: 需要训练

### G-CUT3R — Guided 3D Reconstruction
- **来源**: arXiv 2508.11379, ICLR 2026
- **方法**: 在 CUT3R 上通过 zero convolution 集成 depth/camera prior
- **与我们的区别**: 加先验信息 vs 改 state update 策略
- **威胁程度**: 低。方向不同。

### tttLRM — Test-Time Training for Large Reconstruction Model
- **来源**: arXiv 2602.20160, CVPR 2026
- **方法**: TTT layer 的 fast weights 压缩多视角观测，线性复杂度
- **与我们的区别**: TTT 作为架构（观测→权重）vs TTT3R 的 state update rule
- **意义**: "TTT for 3D" 概念空间更拥挤

### STREAM3R — Scalable Sequential 3D Reconstruction
- **来源**: arXiv 2508.10893
- **方法**: Causal transformer for online incremental 3D reconstruction

### S2GS — Streaming Semantic Gaussian Splatting
- **来源**: arXiv 2603.14232, 2026-03
- **方法**: Strictly causal incremental 3D Gaussian semantic field

---

## 5. 竞争格局总结

### 按是否需要训练分类

| Training-Free | Requires Training |
|---------------|-------------------|
| **IncVGGT** (ICLR 2026) | GRS-SLAM3R |
| **XStreamVGGT** | LONG3R (ICCV 2025) |
| **Test3R** | OnlineX |
| **Ours** | Point3R (NeurIPS 2025) |
| | G-CUT3R (ICLR 2026) |
| | tttLRM (CVPR 2026) |

### 按技术方向分类

| 方向 | 工作 |
|------|------|
| State update gating | GRS-SLAM3R, **Ours**, TTT3R |
| Memory gating/pruning | LONG3R, IncVGGT, XStreamVGGT |
| State 架构改进 | OnlineX (active/stable split), Point3R (spatial pointer) |
| Prior 集成 | G-CUT3R |
| Test-time optimization | Test3R, tttLRM |

### 我们的定位与差异化

**仅存的差异化空间**:
1. **Training-free + plug-and-play** on CUT3R/TTT3R（IncVGGT 也是 training-free 但针对 VGGT 系）
2. **Output-guided state gating**（用模型自身预测的几何一致性指导 state 更新）
3. 不改架构，不改权重

**核心风险**:
- 如果 selective gating 不比 constant dampening 好 → contribution 太薄
- OnlineX 的 insight（高频/低频 state 分离）与我们重叠
- GRS-SLAM3R 的 learned gating 直接竞争

**待确认**:
- S1 video depth + 7scenes 结果：geo gate 在几何任务上是否优于 random dampening
- SIASU v2 cross-token ranking 是否有效
