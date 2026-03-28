# TTT3R — Delta Orthogonalization for Recurrent 3D Reconstruction

## Project Goal
NeurIPS submission. Train-free, inference-time method to address systematic over-update in recurrent 3D reconstruction (CUT3R/TTT3R). 核心方法：Delta Orthogonalization — 将 state update 分解为 drift（重复方向）和 novel（新信息）分量，差异化抑制。

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

**Adaptive 模式** (`--ortho_adaptive`): 根据 per-token drift energy (cos² EMA) 动态调节 α_drift
- `linear`: `α_drift = base + (α_novel - base) × drift_energy`
- `match`: `α_drift = α_novel × drift_energy + base × (1 - drift_energy)`
- `threshold`: per-token 二值切换，drift_energy > 0.5 用 uniform dampening

**Motivation** (A1-A4 分析):
- A1: 所有 scalar adaptive gate 退化为 ~constant 0.33（std≈0.02），无时序动态 → scalar gate 方向不可行
- A2: cos variance vs improvement 无相关（r=-0.13, p=0.63）→ adaptive timing 无意义
- A3: Scalar gate per-scene ~50/50 改善/退化 → 不 robust
- A4: cos(δ_t, δ_{t-1}) TUM≈0.62, ScanNet≈0.77 → drift energy 差异显著（40% vs 60%），需要 dataset-adaptive 策略

## Key Results

### Relpose ATE — TUM（8 sequences）

| Config | ATE ↓ | vs cut3r | vs random |
|--------|-------|----------|-----------|
| cut3r (baseline) | 0.166 | — | — |
| ttt3r | 0.103 | -38.1% | — |
| ttt3r_random (p=0.33) | 0.066 | -60.2% | — |
| ttt3r_momentum_inv_t1 | 0.063 | -61.8% | -4.5% |
| **ttt3r_ortho** | **0.056** | **-66.5%** | **-15.4%** |
| ttt3r_ortho_adaptive (linear) | 0.055 | -66.9% | -16.7% |

### Relpose ATE — ScanNet（96 scenes, 65 valid）

| Config | ATE ↓ | vs cut3r | vs random |
|--------|-------|----------|-----------|
| cut3r (baseline) | 0.817 | — | — |
| ttt3r | 0.406 | -50.3% | — |
| ttt3r_random (p=0.5) | 0.280 | -65.8% | — |
| ttt3r_momentum_inv_t1 | 0.261 | -68.0% | -6.8% |
| **ttt3r_ortho** | **0.492** | -39.8% | +75.7% |
| ttt3r_ortho_adaptive (linear) | 0.358 | -56.2% | +27.9% |
| ttt3r_ortho_adaptive (match) | 运行中 | — | — |
| ttt3r_ortho_adaptive (threshold) | 运行中 | — | — |

**⚠ Ortho 在 ScanNet relpose 上退化**。Adaptive linear 修复部分 (0.492→0.358)，但仍逊于 random/brake。

### Video Depth — Abs Rel ↓

| Config | KITTI | Bonn | Sintel |
|--------|-------|------|--------|
| cut3r | 0.1515 | 0.0990 | 1.0217 |
| ttt3r | 0.1319 (-12.9%) | 0.0997 | 0.9776 (-4.3%) |
| ttt3r_joint | 0.1344 (-11.3%) | 0.0941 (-5.0%) | 0.9173 (-10.2%) |
| **ttt3r_ortho** | **0.1042 (-31.2%)** | **0.0680 (-31.3%)** | **0.4175 (-59.1%)** |

### 3D Reconstruction — 7scenes

| Config | Acc ↓ | Comp ↓ | NC ↑ |
|--------|-------|--------|------|
| cut3r | 0.092 | 0.048 | 0.563 |
| ttt3r | 0.027 (-70.7%) | 0.023 (-52.1%) | 0.581 |
| ttt3r_joint | 0.021 (-77.2%) | 0.022 (-54.2%) | 0.579 |
| **ttt3r_ortho** | **0.026 (-71.7%)** | **0.022 (-54.2%)** | **0.577** |

### Ortho Hyperparameter Sensitivity — TUM

| α_novel | α_drift | β | ATE ↓ | vs default |
|---------|---------|------|-------|------------|
| 0.5 | 0.1 | 0.95 | **0.055** | **-1.5%** |
| **0.5** | **0.05** | **0.95** | **0.056** | **default** |
| 0.5 | 0.2 | 0.95 | 0.056 | +0.1% |
| 0.7 | 0.05 | 0.95 | 0.057 | +2.2% |
| 0.3 | 0.05 | 0.95 | 0.069 | +24.1% |
| 0.5 | 0.05 | 0.9 | 0.076 | +35.5% |
| 0.5 | 0.05 | 0.99 | 0.077 | +38.7% |

α_drift 鲁棒 (0.05-0.2 <2%)，α_novel≥0.5 鲁棒，β=0.95 sweet spot（0.9/0.99 退化 35%+）。

## A4: Delta Direction Analysis — ScanNet vs TUM

| 指标 | TUM (8 scenes) | ScanNet (96 scenes) |
|------|----------------|---------------------|
| **cos(δ_t, δ_{t-1}) mean** | **0.617 ± 0.037** | **0.767 ± 0.037** |
| cos std (intra-scene) | 0.124 ± 0.015 | 0.095 ± 0.011 |
| **drift energy (cos²)** | **0.398 ± 0.041** | **0.598 ± 0.054** |

**关键发现**:
1. **ScanNet cos 远高于 TUM** (0.77 vs 0.62) — ScanNet 室内场景的 state update 方向高度一致
2. **ScanNet drift energy 60% vs TUM 40%** — ScanNet 有 60% 更新能量在 "drift" 方向
3. **ScanNet 的 "drift" 是有用的 refinement** — 室内场景需要在一致方向上持续完善几何，ortho 把这些有用更新当噪声抑制了
4. **TUM drift energy 较低 (40%)** — 动态运动下 drift 确实是重复性 over-update，ortho 分解恰好合适

**Per-scene scatter**: ortho 在 ScanNet 仅 11/65 scenes 改善。cos_mean 与 improvement 弱正相关 (r=0.237, p=0.057)。

脚本: `analysis/a4_delta_direction.py`，结果: `analysis_results/a4_delta_direction/`

## Update Types in model.py

| `model_update_type` | `mask1` (state) | Status |
|---------------------|-----------------|--------|
| `cut3r` | 1.0 (baseline) | baseline |
| `ttt3r` | sigmoid(cross_attn) | baseline |
| `ttt3r_ortho` | delta orthogonalization | **TUM/depth 最优，ScanNet pose 退化** |
| `ttt3r_random` | ttt3r × p (constant) | naive baseline |
| `ttt3r_momentum` | ttt3r × stability brake | 已验证，非最优 |
| `ttt3r_geogate` / `ttt3r_joint` | ttt3r × geo/joint gate | 早期方法 |
| `ttt3r_conf` / `ttt3r_l2gate` | ttt3r × conf/l2 gate | naive baseline |
| Others (spectral, memgate, delta_clip, attn_protect, mem_novelty, brake_geo) | various | 已放弃 |

## Failed Directions (Summary)

- **SIASU v1/v2**: EMA 紧密追踪 → alpha≡0.5; v2 ranking 更差 (0.291 vs 0.283)
- **Geo gate 联合 (brake_geo)**: 两 gate 叠加 over-dampening
- **Delta Clipping**: 限制大更新，旋转场景退化
- **Attention Protection**: 高 attention ≠ 需保护
- **Memory Novelty Gate**: cosine EMA 紧密追踪 (cos>0.99)，≈常数
- **Cross-attention bridge (Route C1)**: Attention 过于 diffuse (entropy 0.914)
- **Non-inverted momentum**: SGD 直觉在 over-update 场景有害
- **Token tracking (Direction C)**: State tokens 不追踪空间语义

## Eval Pipeline

三类评测，脚本在 `eval/` 下：

| 评测类型 | 数据集 | 脚本 | 数据路径 |
|---------|--------|------|---------|
| Camera Pose | ScanNet, TUM, Sintel | `eval/relpose/launch.py` | `data/long_scannet_s3/`, `data/long_tum_s1/` |
| Video Depth | KITTI, Bonn, Sintel | `eval/video_depth/launch.py` | `data/long_kitti_s1/`, `data/long_bonn_s1/` |
| 3D Reconstruction | 7scenes | `eval/mv_recon/launch.py` | — |

```bash
# 双卡并行示例
conda activate ttt3r

# GPU0 — ScanNet relpose
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src accelerate launch --num_processes 1 --main_process_port 29560 \
    eval/relpose/launch.py \
    --weights model/cut3r_512_dpt_4_64.pth --output_dir eval_results/relpose/scannet_s3_1000/<config> \
    --eval_dataset scannet_s3_1000 --size 512 --model_update_type <config> \
    --spectral_temperature 1.0 --geo_gate_tau 2.0 --geo_gate_freq_cutoff 4

# GPU1 — TUM relpose
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src accelerate launch --num_processes 1 --main_process_port 29561 \
    eval/relpose/launch.py \
    --weights model/cut3r_512_dpt_4_64.pth --output_dir eval_results/relpose/tum_s1_1000/<config> \
    --eval_dataset tum_s1_1000 --size 512 --model_update_type <config> \
    --spectral_temperature 1.0 --geo_gate_tau 2.0 --geo_gate_freq_cutoff 4
```

共享参数: `--seed 42 --size 512 --max_frames 200 --num_scannet 10`
并行脚本: `eval/run_parallel_eval.sh`
结果: `eval_results/relpose/<dataset>/<config>/_error_log.txt`

### Paths

- Model: `model/cut3r_512_dpt_4_64.pth`
- 原始数据: `/mnt/sda/szy/research/dataset/` (ScanNet, TUM)
- 本地同步: `rsync -avz 10.160.4.14:/home/szy/research/TTT3R/analysis_results/<exp>/ analysis_results/<exp>/`

### Dataset Notes

- ScanNet: 100 test scenes → 96 预处理（4 empty skip）→ 65 valid（31 GT 含 -inf, evo eigh 不收敛，三配置一致）
- TUM: 8 sequences, 全部成功
- 所有数据集预处理完成，评测 pipeline 正常

## Key Files

| File | Purpose |
|------|---------|
| `src/dust3r/model.py` | 所有 update types, gate methods, LocalMemory |
| `docs/research_progress.md` | 完整研究日志 |
| `docs/related_work.md` | 竞品分析 & 相关工作 |
| `docs/theory_section.tex` | 理论推导 |
| `analysis/a1a2_gate_dynamics.py` | A1/A2 分析脚本 |
| `analysis/a4_delta_direction.py` | A4 delta direction 分析（ScanNet vs TUM） |
| `eval/run_parallel_eval.sh` | 并行评测脚本 |

## Known Issues

1. **Gate state 每帧重置**: `view["reset"]` 返回 `tensor([False])` 非 None → 用 `reset_mask.any()` 判断。已修复三处。
2. **ScanNet 31 scene skip**: GT 含 -inf, evo eigh 不收敛。与原论文一致，不影响公平对比。
3. **`_forward_impl` 扩展**: 已补全所有 update type 支持（含 ttt3r_ortho），与 `inference_step` 对齐。

## Paper Narrative

**叙事**: Over-update 是方向性问题 → Delta Decomposition + 差异化抑制

**Story**:
1. **问题**: Recurrent 3D 的 state update 存在 systematic over-update（常数 dampening ×0.33 即改善 ~60%）
2. **分析**: Scalar adaptive gate 全部退化为常数（A1-A3），但 delta 方向有结构性 drift（A4）
3. **Insight**: 问题不是 "何时更新" 而是 "更新方向的哪部分该保留"
4. **方法**: Delta Orthogonalization — drift/novel 分解 + 差异化抑制
5. **结果**: TUM pose -66.5%, video depth -31~59%, 7scenes Comp -54%; ScanNet pose 退化（A4 解释）

**Contributions**:
1. 揭示 over-update 问题 + scalar gate 退化为常数的证据
2. 发现方向性本质：drift energy 在不同场景差异显著（40%-60%），解释方法适用边界
3. Delta Orthogonalization: train-free, plug-in, video depth SOTA (-31~59%)

## Next Steps

### 运行中
- **[Running] adaptive ortho match/threshold ScanNet** — GPU0/GPU1 并行

### 待办
- **[P0] Adaptive ortho ScanNet 结果分析** — match/threshold 出来后决定最终 adaptive 策略
- **[P1] Inference overhead (S2)** — wall-clock time + peak GPU memory
- **[P1] 理论更新** — drift energy bound, adaptive α 推导
- **[P2] Paper writing** — 基于当前结果开始写 method + experiments
