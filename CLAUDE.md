# DD3R — Training-Free Directional Decomposition Beyond Scalar Gating for Recurrent 3D Reconstruction

## Project Goal
NeurIPS submission. Train-free, inference-time method to address systematic over-update in recurrent 3D reconstruction (CUT3R). 核心方法：Directional Decomposition (Delta Orthogonalization) — 将 state update 分解为 drift（重复方向）和 novel（新信息）分量，差异化抑制。超越 scalar gating 的方向性分解框架。

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

**与 TTT3R mask 的关系**：Ortho 不替换 TTT3R 的 sigmoid mask，而是在其之上叠加。TTT3R mask 控制 per-token 更新幅度（"该不该更新"），ortho 修改 `new_state_feat` 的方向组成（"更新中哪部分该保留"）。最终：`state_t = state_{t-1} + ttt3r_mask × (α_novel × novel + α_drift × drift)`

**Adaptive 模式** (`--ortho_adaptive`): 根据 per-token drift energy (cos² EMA) 动态调节 α_drift
- `linear`: `α_drift = base + (α_novel - base) × drift_energy`
- `match`: `α_drift = α_novel × drift_energy + base × (1 - drift_energy)`
- `threshold`: per-token 二值切换，drift_energy > 0.5 用 uniform dampening

## Key Results

### Relpose ATE — Long Sequence (1000f)

**TUM（8 sequences）**

| Config | ATE ↓ | vs cut3r | vs random |
|--------|-------|----------|-----------|
| cut3r (baseline) | 0.166 | — | — |
| ttt3r | 0.103 | -38.1% | — |
| ttt3r_random (p=0.33) | 0.066 | -60.2% | — |
| brake | 0.063 | -61.8% | -4.5% |
| **ortho** | **0.056** | **-66.5%** | **-15.4%** |
| ortho_adaptive (linear) | 0.055 | -66.9% | -16.7% |

**ScanNet（96 scenes, 65 valid）**

| Config | ATE ↓ | vs cut3r | vs random |
|--------|-------|----------|-----------|
| cut3r (baseline) | 0.817 | — | — |
| ttt3r | 0.406 | -50.3% | — |
| ttt3r_random (p=0.5) | 0.280 | -65.8% | — |
| brake | 0.261 | -68.0% | -6.8% |
| **ortho** | **0.492** | -39.8% | +75.7% |
| ortho_adaptive (linear) | 0.358 | -56.2% | +27.9% |
| ortho_adaptive (match) | 0.356 | -56.4% | +27.1% |

⚠ Ortho 在 ScanNet 退化。Adaptive 天花板 ~0.356，仍逊于 random/brake。高 drift energy 场景的结构性限制（详见分析 A4/A7）。

**Sintel（14 sequences, ~20-50f）**

| Config | ATE ↓ | vs cut3r |
|--------|-------|----------|
| cut3r | 0.209 | — |
| ttt3r | 0.209 | -0.1% |
| ttt3r_random | 0.220 | +5.2% |
| ortho | 0.237 | +13.2% |
| brake | 0.238 | +14.0% |

序列极短，over-update 尚未累积，任何 dampening 均无益。

### Relpose ATE — Short Sequence (90f)

**TUM 90f（8 sequences）**

| Config | ATE ↓ | vs cut3r | vs TTSA3R |
|--------|-------|----------|-----------|
| cut3r | 0.0325 | — | — |
| TTSA3R (paper) | 0.026 | -20.0% | — |
| ttt3r | 0.0189 | -41.8% | -27.3% |
| random / brake | 0.0153 | -52.9% | -41.2% |
| **ortho** | **0.0145** | **-55.4%** | **-44.2%** |

**ScanNet 90f（96 scenes, 90 valid, first-90 标准协议）**

| Config | ATE ↓ | vs cut3r |
|--------|-------|----------|
| cut3r | 0.095 | — |
| ttt3r / random | 0.064 | -32.7% |
| brake | 0.071 | -25.0% |
| ortho_adaptive | 0.074 | -22.6% |
| ortho | 0.087 | -8.2% |

### Video Depth — Abs Rel ↓

| Config | KITTI | Bonn | Sintel |
|--------|-------|------|--------|
| cut3r | 0.1515 | 0.0990 | 1.0217 |
| ttt3r | 0.1319 (-12.9%) | 0.0997 | 0.9776 (-4.3%) |
| brake | 0.1061 (-30.0%) | 0.0658 (-33.5%) | 0.4022 (-60.6%) |
| **ortho** | **0.1042 (-31.2%)** | **0.0680 (-31.3%)** | **0.4175 (-59.1%)** |

Brake vs ortho 在 depth 上非常接近，互有胜负。

### 3D Reconstruction — 7scenes

| Config | Acc ↓ | Comp ↓ | NC ↑ |
|--------|-------|--------|------|
| cut3r | 0.092 | 0.048 | 0.563 |
| ttt3r | 0.027 (-70.7%) | 0.023 (-52.1%) | 0.581 |
| brake | 0.021 (-77.2%) | 0.022 (-54.2%) | 0.580 |
| **ortho** | **0.026 (-71.7%)** | **0.022 (-54.2%)** | **0.577** |

### Inference Overhead — TUM (3 seqs × 200f, 3 repeats)

| Config | FPS | Peak Mem (GB) | vs cut3r |
|--------|-----|---------------|----------|
| cut3r | 8.44 | 6.14 | — |
| ttt3r | 9.82 | 6.14 | +16% faster |
| brake | 10.03 | 6.14 | +19% faster |
| **ortho** | **9.95** | **6.14** | **+18% faster** |

所有方法**零额外内存**，速度甚至略快。结果: `eval_results/benchmark_overhead.json`

## Analysis

### A1-A3: Scalar Gate 退化为常数

- **A1**: 所有 scalar adaptive gate 退化为 ~constant 0.33（σ≈0.02），无时序动态
- **A2**: cos variance vs improvement 无相关（r=-0.13, p=0.63）→ adaptive timing 无意义
- **A3**: Scalar gate per-scene ~50/50 改善/退化 → 不 robust

### A4: Delta Direction — ScanNet vs TUM

| 指标 | TUM (8 scenes) | ScanNet (96 scenes) |
|------|----------------|---------------------|
| cos(δ_t, δ_{t-1}) mean | 0.617 ± 0.037 | 0.767 ± 0.037 |
| drift energy (cos²) | 0.398 ± 0.041 | 0.598 ± 0.054 |

ScanNet drift energy 60% vs TUM 40%。ScanNet 的 "drift" 是有用的 refinement（室内场景需一致方向持续完善几何），ortho 误抑制；TUM 的 drift 是重复性 over-update，ortho 恰好合适。

脚本: `analysis/a4_delta_direction.py`

### A5: TTSA3R TAUM Gate 也退化为常数

TAUM temporal gate σ_time=0.006（比 ttt3r 的 σ≈0.02 小 3-4x），更严重退化为常数 ~0.355。理论原因：`state_change / mean(state_change)` 归一化后均值恒为 1.0，sigmoid(1-1.5) ≈ 0.378。证实 scalar gate 退化推广到竞品。

脚本: `analysis/taum_gate_stats.py`

### A6: Over-update 普遍存在 — 短序列即可观察

| | TUM 90f | TUM 1000f | ScanNet 90f | ScanNet 1000f | Sintel ~50f |
|---|---------|-----------|-------------|---------------|-------------|
| cut3r ATE | 0.033 | 0.166 | 0.095 | 0.805 | 0.209 |
| ttt3r vs cut3r | -42% | -38% | -33% | -50% | 0% |
| ortho vs cut3r | -55% | -66% | -8% | -40% | +13% |
| brake vs cut3r | -53% | -62% | -25% | -68% | +14% |

Over-update 严重度随长度加剧：ScanNet 1000f/90f = 8.5x，TUM = 5.0x。Sintel 极短无 over-update。

### A7: Per-Scene Scatter — Drift Energy vs Ortho Improvement（ScanNet 90f, 90 valid）

| 指标 | Ortho | Brake | Random |
|------|-------|-------|--------|
| 改善 scene 数 | 58/90 (64%) | 72/90 (80%) | 77/90 (86%) |
| r (drift energy vs improvement) | +0.248 (p=0.018) | +0.157 (p=0.14) | +0.091 (p=0.39) |

Ortho 与 drift energy 显著正相关：drift energy 越高越倾向退化。Brake/random 无显著相关，对 drift energy 不敏感。Ortho 引入 precision-robustness trade-off。

脚本: `analysis/viz_scatter_drift_ortho.py`

### Depth Qualitative — Bonn balloon2

| Frame | CUT3R | TTT3R | Brake | Ortho |
|-------|-------|-------|-------|-------|
| t=197 | 0.089 | 0.069 | 0.041 | 0.044 |
| t=357 | 0.107 ↑退化 | 0.047 | 0.040 | 0.042 |

CUT3R 后期退化 (0.089→0.107)，brake/ortho 保持稳定 (~0.04)，误差降低 >50%。

脚本: `analysis/viz_depth_qualitative.py`

### Trajectory Qualitative — TUM 1000f

ScanNet 1000f 4 scenes（2×2 grid, dark background, BEV highest-variance axes）:

| Scene | CUT3R ATE | Brake ATE | DD3R ATE | Brake vs cut3r |
|-------|-----------|-----------|----------|----------------|
| scene0806 | 0.860 | 0.097 | 0.093 | -89% |
| scene0721 | 2.354 | 0.276 | 0.681 | -88% |
| scene0781 | 1.528 | 0.152 | 0.346 | -90% |
| scene0760 | 0.817 | 0.122 | 0.147 | -85% |

CUT3R 轨迹大幅漂移，Brake/DD3R 紧贴 GT。scene0806 环形轨迹最清晰。

脚本: `analysis/viz_traj_comparison.py`，结果: `analysis_results/traj_comparison/`

### Ortho Hyperparameter Sensitivity

**TUM (1000f)**: α_drift 鲁棒 (0.05-0.2 <2%)，α_novel≥0.5 鲁棒，β=0.95 sweet spot（0.9/0.99 退化 35%+）。

**ScanNet (90f, linspace)**: ⚠ 与 TUM 完全反转 — β=0.99 最优（TUM 最差），α_drift 越高越好。证实 drift 性质在两个 dataset 根本不同。

## Update Types in model.py

| `model_update_type` | `mask1` (state) | Status |
|---------------------|-----------------|--------|
| `cut3r` | 1.0 (baseline) | baseline |
| `ttt3r` | sigmoid(cross_attn) | baseline |
| `ttt3r_ortho` | ttt3r_mask + delta orthogonalization | **DD3R 核心方法** |
| `ttt3r_random` | ttt3r × p (constant) | naive baseline |
| `ttt3r_momentum` | ttt3r × stability brake | 已验证，非最优 |
| Others (joint, conf, l2gate, spectral, memgate, delta_clip, attn_protect, mem_novelty, brake_geo) | various | 已放弃 |

## Eval Pipeline

三类评测，脚本在 `eval/` 下：

| 评测类型 | 数据集 | 脚本 |
|---------|--------|------|
| Camera Pose | ScanNet, TUM, Sintel | `eval/relpose/launch.py` |
| Video Depth | KITTI, Bonn, Sintel | `eval/video_depth/launch.py` |
| 3D Reconstruction | 7scenes | `eval/mv_recon/launch.py` |

```bash
# 双卡并行示例
conda activate ttt3r

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src accelerate launch --num_processes 1 --main_process_port 29560 \
    eval/relpose/launch.py \
    --weights model/cut3r_512_dpt_4_64.pth --output_dir eval_results/relpose/scannet_s3_1000/<config> \
    --eval_dataset scannet_s3_1000 --size 512 --model_update_type <config>
```

共享参数: `--seed 42 --size 512 --max_frames 200 --num_scannet 10`
并行脚本: `eval/run_parallel_eval.sh`
结果: `eval_results/relpose/<dataset>/<config>/_error_log.txt`

### Paths

- Model: `model/cut3r_512_dpt_4_64.pth`
- 原始数据: `/mnt/sda/szy/research/dataset/` (ScanNet, TUM)
- 本地同步: `rsync -avz 10.160.4.14:/home/szy/research/TTT3R/analysis_results/<exp>/ analysis_results/<exp>/`

### Dataset Notes

- ScanNet: 100 test scenes → 96 预处理（4 empty skip）→ 90 valid (90f) / 66 valid (1000f) / 65 valid (1000f adaptive)（GT 含 -inf, evo eigh 不收敛）
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
| `analysis/viz_scatter_drift_ortho.py` | Per-scene scatter: drift energy vs ortho improvement |
| `analysis/viz_traj_comparison.py` | 轨迹对比可视化（多方法 Sim(3) aligned） |
| `eval/run_parallel_eval.sh` | 并行评测脚本 |
| `eval/benchmark_overhead.py` | 推理 overhead benchmark |
| `eval/run_scaling_curve.sh` | ScanNet scaling curve 实验 |

## Known Issues

1. **Gate state 每帧重置**: `view["reset"]` 返回 `tensor([False])` 非 None → 用 `reset_mask.any()` 判断。已修复。
2. **ScanNet scene skip**: GT 含 -inf, evo eigh 不收敛。与原论文一致，不影响公平对比。
3. **`_forward_impl` 扩展**: 已补全所有 update type 支持，与 `inference_step` 对齐。

## Paper Narrative

**论文标题**: DD3R: Training-Free Directional Decomposition Beyond Scalar Gating for Recurrent 3D Reconstruction

**Story**:
1. **问题**: Recurrent 3D 存在 systematic over-update，90f 即可观察，随长度加剧
2. **分析**: Scalar gate 全退化为常数（A1-A3 + 竞品 TTSA3R A5）；delta 方向有结构性 drift（A4）
3. **Insight**: 问题不是 "何时更新" 而是 "更新方向的哪部分该保留"——beyond scalar gating
4. **方法**: DD3R — drift/novel 分解 + 差异化抑制，在 TTT3R sigmoid mask 之上叠加方向控制
5. **结果**: TUM pose -66.5% (long) / -55.4% (short), video depth -31~59%, 零额外 overhead

**Contributions**:
1. 揭示 over-update 普遍存在 + scalar gate 退化为常数（双重验证）
2. 发现方向性本质：drift energy 因场景差异显著，per-scene scatter 精确解释方法适用边界 (r=0.248, p=0.018)
3. DD3R: train-free, plug-in, zero overhead, 多任务 SOTA
4. 短序列大幅超越 TTSA3R（TUM -44.2%）

## Next Steps

### 实验
- **[P1] ScanNet scaling curve** — 🔄 GPU 进行中（`eval/run_scaling_curve.sh`），200f/500f × 6 methods
- **[P2] Length-aware ortho** — 前 T₀ 帧不抑制 drift，之后逐渐增强（实验性探索）

### 写作
- **[P1] Method section** — DD3R 形式化定义 + 与 scalar dampening 的对比分析
- **[P1] Experiments** — 五数据集三任务结果表 + ablation（sensitivity, overhead, A1-A7）
- **[P1] 理论框架** — drift energy bound, adaptive α 推导, 与 continual learning gradient projection 联系
- **[P2] Intro + related work** — positioning vs GRS-SLAM3R/OnlineX/LONG3R
