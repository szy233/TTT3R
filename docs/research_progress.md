# TTT3R — 频域引导状态更新研究进展

## 项目概述
TTT3R 基于 CUT3R/TTT3R 循环重建模型，目标投稿 **NeurIPS**。
核心贡献：**频域引导的 state & memory 选择性更新框架**（train-free，轻量推理阶段）

---

## 三层框架

| Layer   | 级别       | 功能                         | 状态       |
|---------|-----------|------------------------------|-----------|
| Layer 1 | 帧级别     | 频域帧筛选，跳过冗余帧            | 已验证 ✅   |
| Layer 2 | Token 级别 | SIASU: 频谱调制 state token 更新 | 待重跑 🔄  |
| Layer 3 | 记忆级别   | B2: spectral_change 门控 memory 写入 | 已实现，待验证 🔄 |

---

## 实验进展

### Exp 0 — Confidence 校准分析
CUT3R confidence 校准优于 TTT3R（r = −0.276 vs −0.218，10个ScanNet场景）。
TUM 动态场景下 confidence 完全失效（TTT3R r ≈ −0.07）→ 需要额外信号。

### Exp 1 — State Token 频谱分析
**脚本**：`analysis/spectral_analysis.py`
帧级别频谱特征与深度误差相关性弱（r ≈ 0.05–0.18）。Confidence 仍是最强预测信号（ScanNet r ≈ −0.38）。

### Exp 2 — Confidence 门控消融（ttt3r_conf）
**脚本**：`analysis/conf_gated_ablation.py`
结果：<1% 改善，校准反而变差（正反馈循环）。**方案放弃**。

### Exp 3 — Layer 1：频域帧筛选 ✅
**脚本**：`analysis/batch_frame_novelty.py`
**指标**：spectral_change = 帧间差的低频能量（LFE），自适应 EMA 阈值过滤冗余帧。

**冗余帧定义**：LFE(δ_t) < τ × EMA(LFE)，即该帧未带来新的结构/几何信息。

**实验证据**：r(spectral_change, state_oscillation) = +0.384（CUT3R, ScanNet, p<0.001）

**结果（18场景，10 ScanNet + 8 TUM，公平评估）**：

| 指标               | CUT3R          | TTT3R          |
|-------------------|---------------|---------------|
| Skip rate (ScanNet)| 35.3% ± 13.5% | 35.3% ± 13.5% |
| r(sc, osc) ScanNet | **+0.384 ± 0.018** | +0.191 ± 0.034 |
| Depth err (ScanNet) | −1.3%         | **−3.1%**      |
| Depth err (TUM)    | +0.9%         | −0.7%          |

> 注：公平评估 = 在相同 kept_indices 上对比全序列与过滤序列深度误差。

### Exp 4 — 帧变化指标对比
**脚本**：`analysis/metric_comparison.py`

| 指标                  | r(osc) ScanNet | err_chg ScanNet |
|----------------------|---------------|----------------|
| spectral_change（低频）| +0.382        | −4.7%          |
| l2_change（基线）      | +0.383        | −4.3%          |
| high_freq_change     | +0.218        | −0.8%          |
| mid_freq_change      | +0.133        | −0.7%          |

**关键发现**：低频 > 高频（验证频域分解意义），但 spectral_change ≈ l2_change。
原因：自然图像服从 1/f² 功率谱，帧级别 L2 已由低频主导。
→ 频域独特贡献需在 **token 级别**或 **memory 管理**层面体现。

### Exp 5 — Layer 2 SIASU（待重跑）
`update_type = ttt3r_spectral / cut3r_spectral`，per-token EMA 高频能量调制更新强度。
**Bug 修复**：running_energy 初始化为 0 → state 冻结。已改为 warm-start。消融结果待重跑。
**脚本**：`analysis/spectral_ablation.py`

### Exp 6 — Direction C：动态感知 Token 更新 ❌ 失败
**脚本**：`analysis/dynamic_token_analysis.py`
**假设**：state token 高频轨迹追踪动态物体，低频追踪静态结构。

**验证结果**：
- Walking 场景：temporal r = −0.024, spatial r = −0.003
- Static 场景：temporal r = −0.383（反向！）

**结论**：State token 不在空间上追踪语义内容。假设不成立，方向放弃。

---

## 当前方向

### B2：Memory Gate — spectral_change 门控 memory 写入 🔄 已实现
**动机**：pose_retriever 的 memory 目前每帧无条件写入（`update_mem()`），无关键帧选择。
冗余帧写入 memory 稀释有效信息。

**方案**：
- 用 `spectral_change`（帧间低频能量）决定是否写入 memory
- 高变化帧 = 关键帧 → g_mem → 1.0（正常写入）
- 低变化帧 = 冗余帧 → g_mem → 0.0（跳过写入）

**实现**：
```
g_mem = sigmoid(τ × (sc / (skip_ratio × ema_sc) − 1))
update_mask2 = update_mask × g_mem
mem = new_mem × update_mask2 + mem × (1 − update_mask2)
```

**新增 update_type**：
- `cut3r_memgate`：state 同 cut3r，memory 加 spectral gate
- `ttt3r_memgate`：state 同 ttt3r，memory 加 spectral gate

**超参数**：`mem_gate_tau=3.0`, `mem_gate_skip_ratio=0.5`, `mem_gate_ema_gamma=0.95`

**消融脚本**：`analysis/memgate_ablation.py`

**状态**：代码已实现，消融实验待运行。

### B3：几何一致性 Gate（待设计）
用前后帧 3D 预测一致性判断 state 更新质量，作为 state 更新的额外门控信号。

---

## LocalMemory (pose_retriever) 机制

```
class LocalMemory:
    mem:          [1, 256, 1536]  — 256 个 memory slot (key 768 + value 768)
    write_blocks: 2× DecoderBlock — cross-attention 写入
    read_blocks:  2× DecoderBlock — cross-attention 读取

update_mem(mem, global_img_feat, out_pose_feat):
    → 用 mem 作 query，新帧特征作 context → cross-attention soft write

inquire(query, mem):
    → 用 [global_img_feat, masked_token] 作 query → cross-attention 从 mem 读出 pose feat
```

当前：每帧无条件调用 `update_mem()`。
B2 改进：低 spectral_change 帧跳过 `update_mem()`。

---

## 数据集

| 数据集      | 服务器路径                                     | 深度 scale | 特点        |
|------------|----------------------------------------------|-----------|------------|
| ScanNet    | `/home/szy/research/dataset/scannetv2/`       | 1000      | 室内静态     |
| TUM-dynamics| `/home/szy/research/dataset/tum/`            | 5000      | 动态人物，8序列 |

---

## 代码结构

| 文件                                  | 功能                                    |
|--------------------------------------|----------------------------------------|
| `src/dust3r/model.py`                | 核心模型，所有 update type 实现             |
| `analysis/spectral_analysis.py`      | State token FFT 分析                     |
| `analysis/batch_frame_novelty.py`    | Layer 1 批量验证                          |
| `analysis/metric_comparison.py`      | 帧变化指标对比                             |
| `analysis/dynamic_token_analysis.py` | Direction C 验证（已失败）                  |
| `analysis/spectral_ablation.py`      | SIASU 消融实验                            |
| `analysis/memgate_ablation.py`       | B2 Memory Gate 消融实验                   |

---

## 关键修复记录

1. **SIASU warm-start bug**：`running_energy` 初始化 0 → ratio≈20 → α≈0 → state 冻结。改为首次调用时 warm-start。
2. **build_views None error**：`to_gpu()` 对 None 调 clone → 改用 `torch.tensor(True/False)`。
3. **TUM depth NaN**：stem 匹配失败 → 改为 `load_tum_associations()` 时间戳匹配。
4. **评估偏差**：全序列 vs 过滤序列比较不公平 → 改为相同 `kept_indices` 上对比。

---

## 待办

- [ ] 运行 B2 memory gate 消融实验 (`memgate_ablation.py`)
- [ ] 重跑 SIASU 消融（warm-start 修复后）
- [ ] 设计 B3 几何一致性 gate
- [ ] 多数据集联合评估
- [ ] 论文 outline 起草
