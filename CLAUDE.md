# DD3R — Training-Free Dampening for Recurrent 3D Reconstruction: Diagnosing and Fixing Systematic Over-Update

## Project Goal
NeurIPS submission. **Analysis-driven paper**：诊断 recurrent 3D reconstruction 中的 systematic over-update 问题，证明现有 scalar gate 全部失效，并提供从简单到精细的 dampening 方案。核心价值在于 **问题发现与分析洞察**，方法是分析的自然推论。

## Paper Narrative

**核心论点**：Recurrent 3D reconstruction 的性能退化源于 systematic over-update，这是 dominant error source。现有 scalar gating 方案（TTT3R, TTSA3R）全部退化为常数，无法解决。连最简单的常数 dampening 都能带来 30-68% 改善，证明问题的严重性。方向性分析进一步揭示 drift 的结构性差异，指导更精细的 dampening 策略。

**Story arc**:
1. **Problem**: Recurrent 3D 存在 systematic over-update，90f 即可观察，随长度加剧（ScanNet 8.5x, TUM 5.0x）
2. **Failed fixes**: 现有 scalar gate 全退化为常数（σ≈0.02 / 0.006），sigmoid 无法产生有意义的时序动态（A1-A3, A5）
3. **Severity**: 哪怕最 naive 的常数 dampening 都大幅改善（30-68%）→ over-update 确实是瓶颈，不是次要因素
4. **Deeper understanding**: Delta 方向有结构性 drift，但性质因场景而异（A4: TUM drift energy 0.40 = 重复 over-update, ScanNet 0.60 = 有用 refinement）
5. **Framework**: DD3R 提供 dampening spectrum — constant → brake → ortho，按场景特性选择

**Contributions**:
1. **C1 Problem Discovery**: 揭示 recurrent 3D 中 systematic over-update 的存在与严重性，随序列长度加剧
2. **C2 Diagnosis**: 证明现有 scalar gate（TTT3R + TTSA3R）全退化为常数，从理论和实验双重验证
3. **C3 Severity Evidence**: 常数 dampening 即可大幅改善（30-68%），证明 over-update 是 dominant failure mode
4. **C4 Directional Understanding**: 发现 drift energy 因场景差异显著，精确解释方法适用边界（r=0.248, p=0.018）
5. **C5 Practical Framework**: DD3R — training-free, plug-in, zero overhead 的 dampening spectrum

**关键叙事策略**：random ≈ brake 不是弱点，而是证据 — "even random works" 恰恰证明 over-update 是 dominant error source。论文定位为 **analysis + insight paper with simple effective fixes**，而非 complex method paper。

## Architecture Overview

Model: `src/dust3r/model.py`, class `ARCroco3DStereo`. Recurrent processing:
1. Encode frame → `feat_i`
2. `_recurrent_rollout(state_feat, feat_i)` → `new_state_feat`, `dec`
3. `pose_retriever.update_mem(mem, feat, pose)` → `new_mem`
4. `_downstream_head(dec)` → `res` (pts3d, conf)
5. State update: `state_feat = new * mask1 + old * (1-mask1)`
6. Memory update: `mem = new_mem * mask2 + mem * (1-mask2)`

`mask1` is where our method is applied.

## Methods

DD3R 框架下三个 operating point，共享基座：TTT3R sigmoid mask 控制 per-token 更新幅度。

### Constant Dampening (`ttt3r_random`)
```python
update_mask1 = ttt3r_mask × p    # p=0.33~0.5, 全局常数
```
最简 baseline。不需要历史信息，零计算开销。验证 over-update 严重性的关键证据。

### Brake — Stability Brake (`ttt3r_momentum`)
```python
delta_t = new_state - state
cos = cosine_similarity(delta_t, delta_{t-1})   # 相邻帧方向一致性
g_brake = sigmoid(-tau × cos)                    # cos高 → gate低 → 抑制
update_mask1 = ttt3r_mask × g_brake
```
**逻辑**：连续更新方向一致 → state 在漂移 → 减速。方向不一致 → 新信息 → 放行。
**特点**：per-token scalar gate，只看相邻两帧，无方向分解。**全场景 robust**。

### Ortho — Delta Orthogonalization (`ttt3r_ortho`)
```python
delta = new_state - state
delta_dir = normalize(delta)
drift_dir = EMA(drift_dir, delta_dir, β=0.95)    # 长期漂移方向（unit vector）
drift_dir = normalize(drift_dir)

drift = proj(delta, drift_dir)                     # 平行于漂移方向 → 强抑制 (α_drift=0.05)
novel = delta - drift                              # 正交分量（新信息）→ 保留 (α_novel=0.5)
updated = state + α_novel × novel + α_drift × drift
update_mask1 = ttt3r_mask   # ortho 修改 new_state_feat 本身，不改 mask
```
**逻辑**：用 EMA 追踪 delta 主方向，将更新分解为"重复漂移"和"新信息"，差异化抑制。
**特点**：directional decomposition，累积历史，per-token per-dimension。**低 drift energy 场景精度最高**。

### 三者关系
- Constant: 不区分时序、不区分方向 → 最 robust 但最粗糙
- Brake: 区分时序（相邻帧 cos）、不区分方向 → robust + 轻度自适应
- Ortho: 区分方向（drift vs novel 分解）→ 最精细但对 drift 性质敏感

## Key Results

### Relpose ATE — Long Sequence (1000f)

**TUM（8 sequences）**

| Config | ATE ↓ | vs cut3r | 角色 |
|--------|-------|----------|------|
| cut3r (baseline) | 0.166 | — | |
| ttt3r | 0.103 | -38.1% | existing method |
| constant (p=0.33) | 0.066 | -60.2% | severity evidence |
| brake | 0.063 | -61.8% | robust default |
| **ortho** | **0.056** | **-66.5%** | precision variant |

**ScanNet（96 scenes, 65 valid）**

| Config | ATE ↓ | vs cut3r | 角色 |
|--------|-------|----------|------|
| cut3r (baseline) | 0.817 | — | |
| ttt3r | 0.406 | -50.3% | existing method |
| constant (p=0.5) | 0.280 | -65.8% | severity evidence |
| **brake** | **0.261** | **-68.0%** | robust default |
| ortho | 0.492 | -39.8% | ⚠ 退化（drift energy 高） |

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

**ScanNet 90f（96 scenes, 90 valid, first-90 标准协议）**

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

Brake vs ortho 在 depth 上非常接近，互有胜负。

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

### A1-A3: Scalar Gate 退化为常数
- **A1**: TTT3R 所有 scalar adaptive gate 退化为 ~constant 0.33（σ≈0.02），无时序动态
- **A2**: cos variance vs improvement 无相关（r=-0.13, p=0.63）→ adaptive timing 无意义
- **A3**: Scalar gate per-scene ~50/50 改善/退化 → 不 robust

### A4: Delta Direction — ScanNet vs TUM
| 指标 | TUM (8 scenes) | ScanNet (96 scenes) |
|------|----------------|---------------------|
| cos(δ_t, δ_{t-1}) mean | 0.617 ± 0.037 | 0.767 ± 0.037 |
| drift energy (cos²) | 0.398 ± 0.041 | 0.598 ± 0.054 |

ScanNet drift energy 60% vs TUM 40%。ScanNet 的 "drift" 是有用 refinement，ortho 误抑制；TUM 的 drift 是 over-update，ortho 恰好合适。

### A5: TTSA3R TAUM Gate 也退化为常数
TAUM temporal gate σ_time=0.006（比 ttt3r 的 σ≈0.02 小 3-4x），更严重退化为常数 ~0.355。
理论原因：`state_change / mean(state_change)` 归一化后均值恒为 1.0，sigmoid(1-1.5) ≈ 0.378。

### A6: Over-update 普遍存在 — 短序列即可观察
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

Ortho 与 drift energy 显著正相关：drift energy 高 → 退化。Brake/constant 无显著相关，对 drift energy 不敏感。

### Qualitative
- **Depth** (Bonn balloon2): CUT3R 后期退化 (0.089→0.107)，brake/ortho 保持稳定 (~0.04)
- **Trajectory** (ScanNet 1000f): CUT3R 轨迹大幅漂移，brake/ortho 紧贴 GT（ATE 降低 85-90%）

### Ortho Hyperparameter Sensitivity
- **TUM**: α_drift 鲁棒 (0.05-0.2 <2%)，α_novel≥0.5 鲁棒，β=0.95 sweet spot
- **ScanNet**: 与 TUM 完全反转 — β=0.99 最优，α_drift 越高越好。证实 drift 性质根本不同。

### Steep Adaptive（Ablation）
Ortho-Brake 统一框架：用 drift energy 控制 selectivity，高 drift energy → 退化为 brake-like。
- TUM 1000f: γ=2 ATE=0.059, γ=3 ATE=0.064（从 ortho 0.056 向 brake 0.063 滑动）
- ScanNet: 🔄 实验进行中

验证 brake 和 ortho 在同一 spectrum 上，drift energy 是核心控制变量。

## Update Types in model.py

| `model_update_type` | `mask1` (state) | 框架角色 |
|---------------------|-----------------|----------|
| `cut3r` | 1.0 (baseline) | baseline |
| `ttt3r` | sigmoid(cross_attn) | existing method |
| `ttt3r_random` | ttt3r × p (constant) | constant dampening |
| `ttt3r_momentum` | ttt3r × stability brake | **brake** |
| `ttt3r_ortho` | ttt3r_mask + delta orthogonalization | **ortho** |
| Others (joint, conf, l2gate, spectral, memgate, delta_clip, attn_protect, mem_novelty, brake_geo) | various | 已放弃 |

## Eval Pipeline

三类评测，脚本在 `eval/` 下：

| 评测类型 | 数据集 | 脚本 |
|---------|--------|------|
| Camera Pose | ScanNet, TUM, Sintel | `eval/relpose/launch.py` |
| Video Depth | KITTI, Bonn, Sintel | `eval/video_depth/launch.py` |
| 3D Reconstruction | 7scenes | `eval/mv_recon/launch.py` |

```bash
conda activate ttt3r
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src accelerate launch --num_processes 1 --main_process_port 29560 \
    eval/relpose/launch.py \
    --weights model/cut3r_512_dpt_4_64.pth --output_dir eval_results/relpose/scannet_s3_1000/<config> \
    --eval_dataset scannet_s3_1000 --size 512 --model_update_type <config>
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
| `analysis/a1a2_gate_dynamics.py` | A1/A2 分析 |
| `analysis/a4_delta_direction.py` | A4 delta direction 分析 |
| `analysis/taum_gate_stats.py` | A5 TTSA3R TAUM gate 分析 |
| `analysis/viz_depth_qualitative.py` | Depth 定性对比可视化 |
| `analysis/viz_scatter_drift_ortho.py` | Per-scene scatter: drift energy vs improvement |
| `analysis/viz_traj_comparison.py` | 轨迹对比可视化（多方法 Sim(3) aligned） |
| `eval/run_parallel_eval.sh` | 并行评测脚本 |
| `eval/benchmark_overhead.py` | 推理 overhead benchmark |
| `eval/run_scaling_curve.sh` | ScanNet scaling curve 实验 |
| `eval/run_steep_eval.sh` | Steep adaptive 实验 |

## Known Issues
1. **Gate state 每帧重置**: `view["reset"]` 返回 `tensor([False])` 非 None → 用 `reset_mask.any()` 判断。已修复。
2. **ScanNet scene skip**: GT 含 -inf, evo eigh 不收敛。与原论文一致，不影响公平对比。
3. **`_forward_impl` 扩展**: 已补全所有 update type 支持，与 `inference_step` 对齐。

## Next Steps

### 实验
- **[P1] ScanNet scaling curve** — 🔄 GPU 进行中，200f/500f × 6 methods
- **[P1] Steep ablation** — 🔄 排队中，TUM g3 完成后跑 ScanNet g2/g3

### 写作
- **[P1] Method section** — DD3R 框架形式化：constant → brake → ortho spectrum
- **[P1] Experiments** — 五数据集三任务结果表 + ablation
- **[P1] 理论框架** — over-update error accumulation, drift energy characterization, scalar gate degeneracy proof
- **[P2] Intro + related work** — positioning: analysis + insight paper, 与 GRS-SLAM3R/OnlineX/LONG3R 的关系
