# TTT3R — Adaptive State Dampening for Recurrent 3D Reconstruction

## Project Goal
NeurIPS submission. Train-free, inference-time adaptive dampening for state updates in recurrent 3D reconstruction (CUT3R/TTT3R). 核心方法：Delta Orthogonalization — 将 state update 分解为 drift（重复方向）和 novel（新信息）分量，差异化抑制。

## Architecture Overview

The model (`src/dust3r/model.py`, class `ARCroco3DStereo`) processes video frames recurrently:
1. Encode frame → `feat_i`
2. `_recurrent_rollout(state_feat, feat_i)` → `new_state_feat`, `dec`
3. `pose_retriever.update_mem(mem, feat, pose)` → `new_mem`
4. `_downstream_head(dec)` → `res` (pts3d, conf)
5. State update: `state_feat = new * mask1 + old * (1-mask1)`
6. Memory update: `mem = new_mem * mask2 + mem * (1-mask2)`

`mask1` and `mask2` are where our adaptive dampening gates are applied.

## Three-Layer Frequency Framework

### Layer 1 — Frame Filtering (validated)
- **Signal**: `LFE(FFT2(RGB_diff))` — low-freq energy of inter-frame RGB difference
- **Action**: Skip frames where LFE < threshold × EMA(LFE)
- **Result**: Skip 35% frames, TTT3R depth -3.1% on ScanNet
- **Code**: `compute_frame_spectral_change()`, `filter_views_by_spectral_change()`

### Layer 2 — Token-Level State Modulation (SIASU) — ❌ 已放弃
- **Signal**: Per-token high-freq residual energy of state trajectory (EMA low-pass → residual)
- **Code**: `_spectral_modulation()`, update types `cut3r_spectral` / `ttt3r_spectral`
- **Status**: **v1 + v2 均失败**
  - v1 公式: `alpha = sigmoid(-τ × (energy / running_energy - 1))`，EMA γ=0.95 紧密追踪 energy → ratio ≈ 1.0 → alpha ≡ 0.5（S4 实验确认：token/frame 方差均为 0）
  - 之前 "L2+ttt3r -8.2%" 的提升**全部来自乘常数 0.5**，非频域信号
  - v2 改用 cross-token ranking（`percentile = energy / max(energy)`），ScanNet ATE = 0.291，比 v1 joint (0.283) 还差，未带来改善
  - **结论**: Token-level spectral modulation 方向不可行，已被 Stability Brake 替代

### Layer 3 — Geometric Consistency Gate (validated, best result)
- **Signal**: `LFE(FFT2(log_depth_diff))` — low-freq energy of log-depth change
- **Action**: Gate state update when depth prediction is geometrically inconsistent
- **Result**: cut3r_geogate -3.5%, ttt3r_geogate -7.2% (vs ttt3r -6.4%)
- **Code**: `_geo_consistency_gate()`, update types `cut3r_geogate` / `ttt3r_geogate`
- **Best config**: τ=2, freq_cutoff=4 (25% bandwidth). Cutoff-insensitive (c2/c4/c8 all ~-3.5%)

## Update Types in model.py

| `model_update_type` | `mask1` (state) | `mask2` (memory) |
|---------------------|-----------------|------------------|
| `cut3r` | 1.0 (baseline) | 1.0 |
| `ttt3r` | sigmoid(cross_attn) | 1.0 |
| `cut3r_spectral` | spectral_modulation α | 1.0 |
| `ttt3r_spectral` | ttt3r × α | 1.0 |
| `cut3r_memgate` | 1.0 | spectral_change gate |
| `ttt3r_memgate` | sigmoid(cross_attn) | spectral_change gate |
| `cut3r_geogate` | geo_consistency gate | 1.0 |
| `ttt3r_geogate` | ttt3r × geo gate | 1.0 |
| `cut3r_joint` | α × geo gate | 1.0 |
| `ttt3r_joint` | ttt3r × α × geo gate | 1.0 |
| `ttt3r_l2gate` | ttt3r × l2_norm_gate | 1.0 | ← naive baseline (implemented)
| `ttt3r_random` | ttt3r × p (constant) | 1.0 | ← naive baseline (implemented, p=0.5)
| `ttt3r_conf` | ttt3r × conf_gate | 1.0 | ← naive baseline (existing)
| `ttt3r_momentum` | ttt3r × momentum_gate | 1.0 | stability brake (inverted cosine)
| `ttt3r_brake_geo` | ttt3r × momentum_gate × geo gate | 1.0 | brake+geo 联合（over-dampening, 已放弃）
| `ttt3r_ortho` | delta orthogonalization | 1.0 | ← **当前最优方法**（drift/novel 分解）
| `ttt3r_delta_clip` | ttt3r × clip(delta) | 1.0 | delta norm clipping（失败）
| `ttt3r_attn_protect` | ttt3r × attn_protect | 1.0 | attention-based protection（失败）
| `ttt3r_mem_novelty` | ttt3r × mem_novelty_gate | 1.0 | memory novelty gate（≈常数，失败）

## Key Experimental Results

### B2 Memory Gate (weak, ~1% improvement)
```
cut3r_mg_t3_sr0.3   -1.75%   (best memgate variant)
ttt3r_mg_t3_sr0.5   -6.25%   (no gain over pure ttt3r -6.40%)
```
Memory's soft cross-attention write already handles redundancy. Direction deprioritized.

### B3 Geometric Consistency Gate (strong)
```
cut3r_geo_t2         -3.83%   (spatial domain, best)
cut3r_geo_t2_c4      -3.52%   (frequency domain)
ttt3r_geo_t3         -7.41%   (spatial domain, best overall)
ttt3r_geo_t2_c4      -7.16%   (frequency domain)
```

### Joint Ablation (L23+ttt3r is best)
```
cut3r (baseline)     0.0745   —
ttt3r (baseline)     0.0697   -6.4%
L1+ttt3r             0.0700   -6.0%
L2+ttt3r             0.0684   -8.2%
L3+ttt3r             0.0692   -7.2%
L23+ttt3r            0.0690   -7.5%   ← best combination
L123+ttt3r           0.0699   -6.2%   (L1 conflicts with L2/L3)
L23+cut3r            0.0698   -6.3%   (matches pure ttt3r)
```
L1 frame skipping conflicts with fine-grained L2/L3 modulation. Final method: L23+ttt3r.

### S1 Naive Baseline + Stability Brake 全量对比 (2026-03-27)

**Relpose ATE (Average) — ScanNet（全量 96 scenes, 65 valid）**

| Config | ATE ↓ | vs cut3r | vs ttt3r | vs random |
|--------|-------|----------|----------|-----------|
| cut3r (baseline) | 0.817 | — | — | — |
| ttt3r | 0.406 | -50.3% | — | — |
| **ttt3r_momentum_inv_t1** | **0.261** | **-68.0%** | **-35.6%** | **-6.8%** |
| ttt3r_random (p=0.5) | 0.280 | -65.8% | -31.1% | — |
| ttt3r_joint (v1, alpha≡0.5) | 0.283 | -65.4% | -30.3% | +1.1% |
| ttt3r_joint_fixed | 0.284 | -65.2% | -30.0% | +1.4% |
| ttt3r_joint_siasu_v2 | 0.291 | -64.4% | -28.3% | +3.9% |
| ttt3r_conf | 0.298 | -63.5% | -26.6% | +6.4% |
| ttt3r_momentum_inv_t2 | 0.311 | -62.0% | -23.5% | +11.1% |
| ttt3r_momentum_v2 (non-inv) | 0.345 | -57.8% | -15.1% | +23.2% |
| ttt3r_l2gate_fixed | (running, 50/96) | — | — | — |

**Relpose ATE (Average) — TUM（8 sequences）**

| Config | ATE ↓ | vs cut3r | vs ttt3r | vs random |
|--------|-------|----------|----------|-----------|
| cut3r (baseline) | 0.166 | — | — | — |
| ttt3r | 0.103 | -38.1% | — | — |
| **ttt3r_momentum_inv_t1** | **0.063** | **-61.8%** | **-38.2%** | **-20.3%** |
| ttt3r_joint (v1) | 0.069 | -58.6% | -33.5% | -12.7% |
| ttt3r_joint_fixed | 0.071 | -57.0% | -30.7% | -10.1% |
| ttt3r_conf | 0.073 | -56.1% | -29.1% | -7.6% |
| ttt3r_l2gate_fixed | 0.075 | -54.7% | -26.9% | -5.1% |
| ttt3r_random (p=0.5) | 0.079 | -52.4% | -23.3% | — |
| ttt3r_momentum_inv_t2 | 0.082 | -50.5% | -20.1% | +3.8% |
| ttt3r_momentum_v2 (non-inv) | 0.098 | -40.8% | -4.2% | +24.1% |

**关键发现 (2026-03-27)**:
1. **Stability brake inv_t1 是全场最佳**: ScanNet -6.8% vs random, TUM -20.3% vs random
2. **自适应 dampening 有独立价值**: 不是常数 ×0.5 能解释的，尤其在动态场景 (TUM) 上
3. **τ=1 > τ=2**: 温和调节优于激进调节（inv_t2 在两个数据集上都更差）
4. **Non-inverted momentum 确认有害**: 0.345 (ScanNet), 接近 pure ttt3r
5. **SIASU v2 无效**: 0.291, 比 v1 joint (0.283) 还差
6. **理论与实验一致**: TUM（运动多样性高，cos 方差大）改善 -20%，ScanNet（室内静态，cos 方差小）改善温和 -7%

### Delta Orthogonalization (ttt3r_ortho) — ✅ 当前最优 (2026-03-28)

**方法**: 将 state update delta 分解为 drift（EMA 追踪的重复方向）和 novel（正交新信息）分量，差异化抑制：
```python
delta = new_state_feat - state_feat
drift_dir = EMA(delta_dir, β=0.95)  # 追踪漂移方向
drift_comp = proj(delta, drift_dir)   # 投影到漂移方向
novel_comp = delta - drift_comp       # 正交分量（新信息）
updated = state_feat + α_novel × novel_comp + α_drift × drift_comp
# α_novel=0.5, α_drift=0.05 — 保留新信息，强抑制重复漂移
```

**Relpose ATE — TUM（8 sequences）**

| Config | ATE ↓ | vs cut3r | vs random_p033 |
|--------|-------|----------|----------------|
| cut3r (baseline) | 0.166 | — | — |
| ttt3r | 0.103 | -38.1% | — |
| ttt3r_random (p=0.33) | 0.066 | -60.2% | — |
| ttt3r_momentum_inv_t1 | 0.063 | -61.8% | -4.5% |
| **ttt3r_ortho** | **0.056** | **-66.5%** | **-15.4%** |
| ttt3r_ortho (α_drift=0) | 0.075 | -54.7% | +13.6% |

**Per-sequence breakdown**: 7/8 improved, 1 tied (walking_static)
- Best: sitting_halfsphere 0.070→0.040 (-42.9%), walking_halfsphere 0.086→0.046 (-46.5%)
- Worst: walking_rpy 0.110→0.153 (+39.1%) — 唯一退化

**关键发现 (2026-03-28)**:
1. **Ortho 是全场最佳**: TUM -15.4% vs constant 0.33, -11.1% vs stability brake
2. **α_drift=0 失败 (0.075)**: 漂移方向含有用信号，不能完全丢弃，需小量保留 (0.05)
3. **Over-update 本质是 scale calibration 问题**: A1 分析揭示所有 adaptive gate std≈0.02-0.03，退化为 ~constant 0.33。真正的问题不是 "何时更新"，而是 "更新方向的哪部分该保留"
4. **cos mean=0.7**: 70% 的 delta 能量在重复方向上 → drift 抑制有实质意义

**ScanNet 全量评测**: 运行中（GPU0, 96 scenes）

### A1/A2/A3 Deep Analysis 结果 (2026-03-28)

**A1: Gate Temporal Dynamics**
- Gate std 仅 0.02-0.03（across all adaptive methods），几乎无时序变化
- Gate 值 ~0.33 常数，与 camera motion 无相关性
- **结论**: Stability brake 的 "自适应" 实际退化为常数 dampening，改善来自 scale 而非 timing

**A2: Cosine Variance ↔ Improvement 相关性**
- Pearson r = -0.133, p = 0.625（不显著）
- Var(cos) 与 ATE improvement 无相关
- **结论**: 之前 "TUM Var(cos) > ScanNet 导致 TUM 改善更多" 的解释不成立

**A3: Per-Scene 改善分布**
- Stability brake vs random: 约 50/50 改善/退化，非 consistent improvement
- **结论**: 均值改善但 per-scene 不一致，进一步支持 "scale calibration > adaptive timing"

**S1 video depth + 7scenes 尚未完成**: `mv_recon/launch.py` 不接受 `--eval_dataset` 参数，7scenes baseline 脚本需修复。

**brake_geo 联合实验 (2026-03-27)**:
- ScanNet ATE 0.339（vs inv_t1 0.261, +30%↑），TUM ATE 0.081（vs inv_t1 0.063, +29%↑）
- Geo gate 叠加后 over-dampening，**确认 stability brake 单独使用最优**

### S3 Hyperparameter Sensitivity (2026-03-26)

**ScanNet Relpose ATE (Average), varying spectral_temperature**

| spectral_temperature | ATE ↓ |
|---------------------|-------|
| 0.5 | 0.280 |
| 1.0 | 0.281 |
| 2.0 | 0.286 |
| 4.0 | (running) |

τ 完全无影响，因为 SIASU v1 alpha ≡ 0.5。S3 结果在 v1 下无意义，需等 SIASU v2 后重跑。

### S4 Gate Visualization (2026-03-26)

**3 个 ScanNet scene (scene0707, scene0708, scene0709) 的 gate 统计**

| Component | Mean | Std | Range |
|-----------|------|-----|-------|
| ttt3r_mask | 0.300 | 0.045 | [0.19, 0.39] |
| **SIASU alpha (v1)** | **0.500** | **0.000** | **[0.50, 0.50]** |
| Geo gate g_geo | 0.528 | 0.268 | [0.0006, 1.0] |
| Effective mask | 0.079 | 0.042 | [0.0001, 0.18] |

**诊断**: SIASU v1 alpha 零方差，退化为常数 0.5。ttt3r_mask 均值 0.30（未饱和）。Geo gate 有真实方差（std=0.27），信号 legitimate 但在 relpose 上未优于常数 dampening。

### Failed Directions
- **Direction C (dynamic token tracking)**: State tokens don't track spatial semantics. Walking r=-0.024, static r=-0.383 (reversed). Abandoned.
- **Confidence gating (Exp 2)**: <1% improvement, feedback loop. Abandoned.
- **SIASU v1 + v2**: v1 EMA 紧密追踪 → alpha ≡ 0.5; v2 cross-token ranking ATE 0.291, 比 v1 (0.283) 还差。Token-level spectral modulation 方向整体不可行。
- **Route C1 (cross-attention bridge)**: Decoder cross-attention 太 diffuse（normalized entropy 0.914, cosine sim 0.772），无法将 pixel-space gate 有效传递到 token space。Token gate 退化为 scalar mean(pixel_gate)。
- **原始 momentum gate (non-inverted)**: cos~0.74 → gate~0.80 → 几乎不 dampening → 比常数 0.5 差。SGD momentum 直觉在 over-update 场景下有害。
- **brake_geo (stability brake + geo gate 联合)**: ScanNet 0.339, TUM 0.081，两个 gate 各自 ~0.5 叠加后 over-dampening（effective ~0.25），不如单独 stability brake (0.261/0.063)。
- **Delta Clipping (ttt3r_delta_clip)**: TUM ATE 0.104, 比 ttt3r (0.103) 无改善。Clip 在需要大更新时（旋转场景）适得其反，halfsphere/walking_xyz 严重退化。
- **Attention Protection (ttt3r_attn_protect)**: TUM ATE 0.070, 保护高 attention token 不帮助。假设（频繁关注=重要 token 需保护）不成立。
- **Memory Novelty Gate (ttt3r_mem_novelty)**: TUM ATE 0.066 ≈ constant 0.33。Feature 空间 cosine EMA 紧密追踪（cos>0.99），与 SIASU v1 同样问题。

### Stability Brake (inverted momentum gate) — 已验证但非最优 (2026-03-27)

**方法**: `α_t = σ(-τ·cos(δ_t, δ_{t-1}))` — state 更新方向一致时制动（cos 高→α 小），突变时放行（cos 低→α 大）

**全量结果已确认**（见 S1 对比表）:
- **ScanNet**: inv_t1 ATE 0.261, vs random 0.280 (-6.8%)
- **TUM**: inv_t1 ATE 0.063, vs random 0.079 (-20.3%)
- τ=1 全面优于 τ=2（温和调节更好）
- Non-inverted (momentum_v2) 确认有害（ScanNet 0.345, TUM 0.098）
- 早期 16 scenes 趋势在全量 96 scenes 上完全复现

## Experiment Configs

All experiments share: `--seed 42 --size 512 --max_frames 200 --num_scannet 10`

Server paths:
- Model: `model/cut3r_512_dpt_4_64.pth`
- ScanNet: `/mnt/sda/szy/research/dataset/scannetv2`
- TUM: `/mnt/sda/szy/research/dataset/tum`
- Working dir: `/home/szy/research/TTT3R`

Local paths:
- Working dir: `/Users/shaozhengyu/code/TTT3R`
- Results synced to `analysis_results/` (gitignored)

Sync command: `rsync -avz 10.160.4.14:/home/szy/research/TTT3R/analysis_results/<exp>/ analysis_results/<exp>/`

## Key Files

| File | Purpose |
|------|---------|
| `src/dust3r/model.py` | All update types, gate methods, LocalMemory |
| `analysis/geogate_ablation.py` | B3 geo gate ablation |
| `analysis/memgate_ablation.py` | B2 memory gate ablation |
| `analysis/spectral_ablation.py` | Layer 2 SIASU ablation |
| `analysis/batch_frame_novelty.py` | Layer 1 validation |
| `analysis/metric_comparison.py` | spectral_change vs L2/high/mid freq |
| `analysis/joint_ablation.py` | Three-layer joint ablation |
| `docs/research_progress.md` | Full research log (Chinese) |
| `docs/related_work.md` | 竞品分析 & 相关工作整理 (2026-03-26) |
| `docs/run_experiments.sh` | All experiment commands |
| `analysis/s4_gate_visualization.py` | S4 gate activation 可视化 |
| `analysis/check_cross_attn_sparsity.py` | Cross-attention sparsity 验证（Route C 前提检验） |
| `analysis/a1a2_gate_dynamics.py` | A1/A2 gate dynamics + cos variance 分析 |
| `analysis/a1_outdoor.py` | KITTI/Sintel A1 分析 |
| `analysis/token_gate_variance.py` | Per-token gate 方差分析 |
| `docs/theory_section.tex` | 理论推导（over-update bound, regret bound, optimal τ） |

## Formal Evaluation

### Eval Pipeline

三类标准评测，脚本在 `eval/` 下：

| 评测类型 | 数据集 | 脚本 | 预处理数据路径 |
|---------|--------|------|--------------|
| Camera Pose (relpose) | ScanNet, TUM, Sintel | `eval/relpose/launch.py` | `data/long_scannet_s3/`, `data/long_tum_s1/` |
| Video Depth | KITTI, Bonn, Sintel | `eval/video_depth/launch.py` | `data/long_kitti_s1/`, `data/long_bonn_s1/` |
| 3D Reconstruction | 7scenes | `eval/mv_recon/launch.py` | — |

### 运行方式

对比配置：`cut3r`（baseline）, `ttt3r`, `ttt3r_ortho`（Delta Orthogonalization，当前最优）, `ttt3r_random`（constant dampening baseline）。

```bash
# 双卡并行: GPU0 跑 ScanNet, GPU1 跑 TUM
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

并行脚本: `eval/run_parallel_eval.sh`（nohup 双卡，日志 `eval/gpu0_scannet.log`, `eval/gpu1_tum.log`）

### 预处理

```bash
conda activate ttt3r
python datasets_preprocess/prepare_scannet_local.py   # → data/long_scannet_s3/ (96 scenes, 4 empty skipped from 100 test scenes)
python datasets_preprocess/prepare_tum_local.py       # → data/long_tum_s1/ (8 sequences)
```

原始数据在 `/mnt/sda/szy/research/dataset/`（从根分区迁出）。

### 数据集状态（2026-03-24）

| 数据集 | 原始数据 | 预处理 | 评测状态 |
|--------|---------|--------|---------|
| ScanNet | ✅ `/mnt/sda/szy/research/dataset/scannetv2` (100 test scenes) | ✅ `data/long_scannet_s3/` (96 scenes; 4 empty skipped) | ✅ 完成 (65 valid, 31 GT含-inf skip) |
| TUM | ✅ `/mnt/sda/szy/research/dataset/tum` | ✅ `data/long_tum_s1/` (8 seqs) | ✅ 完成 |
| Sintel | ✅ `data/sintel/` | — (直接使用) | ✅ 完成 |
| Bonn | ✅ `data/long_bonn_s1/` | ✅ 预处理完成 | ✅ 完成 |
| KITTI | ✅ `data/long_kitti_s1/` | ✅ 预处理完成 | ✅ 完成 |
| 7scenes | ✅ 已下载 | ✅ 预处理完成 (18 seqs, 7 scenes) | ✅ 完成 |

结果输出到 `eval_results/relpose/<dataset>/<config>/_error_log.txt`（ATE, RPE trans, RPE rot）。

### Relpose 评测结果（2026-03-24）

**ScanNet（96 scenes 中 65 valid, 31 skip — GT pose 含 -inf 导致 evo Umeyama eigh 不收敛，三配置一致，与原论文行为对齐）**

| Config | ATE (median) ↓ | RPE_t (median) ↓ | RPE_r (median) ↓ |
|--------|----------------|-------------------|-------------------|
| cut3r (baseline) | 0.6713 | 0.0322 | 0.8987 |
| ttt3r | 0.3519 (-47.6%) | 0.0350 | 0.9105 |
| **ttt3r_joint** | **0.2143** (-68.1%) | 0.0449 | 1.0805 |

**TUM（8 sequences，全部成功）**

| Config | ATE (median) ↓ | RPE_t (median) ↓ | RPE_r (median) ↓ |
|--------|----------------|-------------------|-------------------|
| cut3r (baseline) | 0.1641 | 0.0072 | 0.5655 |
| ttt3r | 0.1043 (-36.4%) | 0.0091 | 0.4859 |
| **ttt3r_joint** | **0.0589** (-64.1%) | 0.0103 | 0.4758 |

**分析**: ATE 大幅改善（ScanNet -68%, TUM -64%），RPE_t/RPE_r 略有上升，说明方法显著提升全局轨迹一致性，逐帧相对误差有小幅代价。31 个 Eigenvalue failure 在三配置间一致，不影响公平对比。

### Video Depth 评测结果（2026-03-24）

**Abs Rel ↓**

| Config | KITTI | Bonn | Sintel |
|--------|-------|------|--------|
| cut3r (baseline) | 0.1515 | 0.0990 | 1.0217 |
| ttt3r | 0.1319 (-12.9%) | 0.0997 (+0.7%) | 0.9776 (-4.3%) |
| **ttt3r_joint** | **0.1344** (-11.3%) | **0.0941** (-5.0%) | **0.9173** (-10.2%) |

**δ < 1.25 ↑**

| Config | KITTI | Bonn | Sintel |
|--------|-------|------|--------|
| cut3r (baseline) | 0.8043 | 0.9061 | 0.2377 |
| ttt3r | 0.8653 | 0.9214 | 0.2324 |
| **ttt3r_joint** | 0.8577 | **0.9343** | **0.2472** |

**分析**: ttt3r_joint 在三个数据集上 Abs Rel 全面优于 baseline（KITTI -11.3%, Bonn -5.0%, Sintel -10.2%）。KITTI 上纯 ttt3r 略优于 joint，Bonn 和 Sintel 上 joint 最佳。

### 3D Reconstruction 评测结果（2026-03-25）

**7scenes（18 sequences, 7 scenes, 每 seq 限 200 帧）**

结果路径: `eval_results/video_recon/7scenes_200/<config>/7scenes/logs_all.txt`

| Config | Acc ↓ | Comp ↓ | NC ↑ | NC_med ↑ |
|--------|-------|--------|------|----------|
| cut3r (baseline) | 0.092 | 0.048 | 0.563 | 0.596 |
| ttt3r | 0.027 (-70.7%) | 0.023 (-52.1%) | 0.581 (+3.2%) | 0.625 (+4.9%) |
| **ttt3r_joint** | **0.021** (-77.2%) | **0.022** (-54.2%) | 0.579 (+2.8%) | 0.622 (+4.4%) |

完整指标（mean）：

| Config | Acc ↓ | Comp ↓ | NC1 ↑ | NC2 ↑ | Acc_med ↓ | Comp_med ↓ | NC1_med ↑ | NC2_med ↑ |
|--------|-------|--------|-------|-------|-----------|------------|-----------|-----------|
| cut3r | 0.092 | 0.048 | 0.582 | 0.545 | 0.054 | 0.018 | 0.627 | 0.566 |
| ttt3r | 0.027 | 0.023 | 0.600 | 0.561 | 0.015 | 0.005 | 0.657 | 0.593 |
| ttt3r_joint | 0.021 | 0.022 | 0.594 | 0.565 | 0.009 | 0.004 | 0.646 | 0.598 |

**分析**: ttt3r_joint 在 3D 重建上表现最佳，Accuracy -77.2%, Completeness -54.2%。纯 ttt3r 也有大幅改善（Acc -70.7%）。法向一致性均有提升（NC +2~5%）。

**Bug fix (2026-03-25)**: `_forward_impl()` 原先只支持 `cut3r`/`ttt3r`，`mv_recon/launch.py` 调用 `model(batch)` → `forward()` → `_forward_impl()`，导致 `ttt3r_joint` 报 `Invalid model type`。已补全所有 update type 支持（spectral, geogate, joint 等），与 `inference_step` 路径对齐。日志: `eval/7scenes_recon_joint.log`。

## Known Issues / Fixes Applied
1. **SIASU warm-start**: `running_energy` init 0 → ratio explosion → state frozen. Fixed: warm-start on first call.
2. **TUM depth matching**: Timestamp-based association needed (not stem-based).
3. **Fair evaluation**: Compare full vs filtered on same `kept_indices`.
4. **ScanNet pose 截断**: 根分区满时 `prepare_scannet_local.py` 写 pose 文件被截断（scene0707_00）。已修复重新生成。
5. **ScanNet 31 scene Eigenvalue failure**: GT pose 含 -inf（深度传感器丢失追踪），evo Umeyama `eigh()` 不收敛。与原论文行为一致（同样 skip），不影响公平对比。4 个 scene (0777-0780) .sens 未解压，预处理跳过。
6. **`_forward_impl` 缺少扩展 update type**: 只支持 cut3r/ttt3r，导致 mv_recon 评测 ttt3r_joint 失败。已补全所有类型（spectral, geogate, memgate, joint）并添加 spectral_state/geo_state 的 reset 逻辑。
7. **SIASU v1 alpha ≡ 0.5 退化**: EMA γ=0.95 紧密追踪 energy → ratio ≈ 1.0 → sigmoid(0)=0.5。修复：v2 改用 cross-token ranking（`percentile = energy / max(energy)`），保证 token 间有差异。
8. **`run_baseline_eval.sh` video depth 数据集名错误**: `kitti_1000` → 应为 `kitti_s1_500`，`bonn_1000` → `bonn_s1_500`。Relpose 部分已跑完，video depth/7scenes 需修复后重跑。
9. **Gate state 每帧重置 (CRITICAL, 2026-03-26)**: `reset_mask = view["reset"]` 返回 `tensor([False])`（不是 None），`if reset_mask is not None:` 永远 True → momentum_state/l2_state/spectral_state 每帧重置。修复：`if reset_mask.any():` 包裹 state reset 代码，三处（`_forward_impl`, `forward_recurrent_lighter`, `forward_recurrent_analysis`）。Geo gate 不受影响。
10. **原始 momentum gate 方向错误**: SGD momentum 直觉（cos 高→更新多）在 over-update 场景下有害。反转为 stability brake（`sigmoid(-τ × cos)`）后有效。
11. **SIASU v2 无效 (2026-03-27)**: cross-token ranking ATE 0.291, 比 v1 (0.283) 还差。Token-level spectral modulation 方向整体放弃。
12. **S1 7scenes 脚本错误 (2026-03-26)**: `run_baseline_eval.sh` 中 7scenes 部分传了 `--eval_dataset 7scenes`，但 `mv_recon/launch.py` 不接受该参数。Video depth 部分正常完成。

## Paper Narrative（2026-03-28 更新，Delta Orthogonalization）

### 叙事定位

**不再从频域或 adaptive timing 出发**。A1/A2/A3 分析揭示：所有 scalar adaptive gate（stability brake, geo gate, SIASU）都退化为 ~constant 0.33（std≈0.02），与 camera motion 无相关。真正的问题不是 "何时更新" 而是 "更新的哪个方向该保留"。

新定位：**Over-update 是方向性问题 → Delta Decomposition + 差异化抑制**。

### 核心故事线

**问题**: Recurrent 3D reconstruction 的 state update 存在 systematic over-update。实验证据：
1. 任何常数 dampening（×0.33）都改善 ~60%，说明 over-update 是核心瓶颈
2. TTT3R 的 learned gate 训练在 image pairs 上，未见过 long video temporal dynamics

**关键分析发现**:
- A1: 所有 adaptive scalar gate 退化为 ~constant 0.33（std≈0.02），无时序动态
- A2: cos variance vs improvement 无相关（r=-0.13, p=0.63）→ adaptive timing 无意义
- A3: Scalar gate 的 per-scene consistency ~50%，非 robust improvement
- cos(δ_t, δ_{t-1}) mean≈0.7 → **70% delta 能量在重复方向** → 问题是方向性的

**Insight**: Over-update 不是 scale 问题（scalar gate 解决），而是**方向性漂移**问题。连续帧的 state update 70% 重复，需要方向分解而非 scalar dampening。

**方法**: Delta Orthogonalization — 分解 delta 为 drift + novel 分量
```
drift_dir = EMA(delta_direction, β=0.95)
drift_comp = proj(delta, drift_dir)    # 重复方向 → 强抑制 (α_drift=0.05)
novel_comp = delta - drift_comp        # 新信息 → 保留 (α_novel=0.5)
```
- Train-free, inference-time, 三个超参（α_novel, α_drift, β）
- 直接 plug-in，不改模型结构
- TUM: -15.4% vs constant dampening, -11.1% vs stability brake

**理论支撑** (`docs/theory_section.tex`):
1. Over-update bound: 无 dampening 时误差 O(k²) 增长
2. 方向分解优势: drift 分量占 70% 能量，选择性抑制比 uniform dampening 保留更多新信息
3. α_drift=0 失败证明漂移方向仍含有用信号，需小量保留

### Contribution

1. 揭示 recurrent 3D reconstruction 中 systematic over-update 问题，通过 A1/A2/A3 分析证明 scalar adaptive gate 退化为常数
2. 发现 over-update 的方向性本质：70% delta 能量在 drift 方向，scalar dampening 无法区分 drift vs novel
3. 提出 Delta Orthogonalization（drift/novel 分解 + 差异化抑制），TUM -66.5% vs baseline, -15.4% vs constant dampening

### Deep Analysis（已完成，支撑方法设计）

#### A1: Gate Temporal Dynamics — ✅ 完成
- **结果**: Gate std 0.02-0.03，~constant 0.33，与 camera motion 无相关
- **意义**: 证明 scalar adaptive gate 方向不可行，motivate 方向分解

#### A2: Cosine Variance ↔ Improvement — ✅ 完成
- **结果**: Pearson r=-0.133, p=0.625，不显著
- **意义**: "运动多样性高→自适应改善多" 的假设不成立

#### A3: Per-Scene 改善分布 — ✅ 完成
- **结果**: Brake vs random ~50/50 改善/退化
- **意义**: Scalar gate 改善不 consistent，需要更 principled 的方法

#### A4: Delta 方向分析（支撑 ortho 设计）
- cos(δ_t, δ_{t-1}) mean≈0.7 → 70% 能量在 drift 方向
- α_drift=0 failure → drift 方向非纯 noise，需小量保留
- **意义**: 直接 motivate drift/novel 分解的设计选择

## Supplementary Experiments（2026-03-28 更新）

### Exp S1: Naive Baseline Comparison（relpose 全量完成）

已实现所有 baseline + adaptive methods。**Delta Orthogonalization (ttt3r_ortho) 是当前最优**。

**完成状态**:
- Relpose 全量: stability brake, random, conf, l2gate, brake_geo 全部完成
- ttt3r_ortho TUM 完成 (ATE 0.056), ScanNet 运行中
- Video depth + 7scenes 过夜实验已完成（`eval/overnight.log`）

### Exp S2: Inference Overhead（必须）

Train-free 是卖点，需要证明 overhead negligible。

- 每个配置跑 wall-clock time 和 peak GPU memory
- 配置: `cut3r`, `ttt3r`, `ttt3r_joint`, `ttt3r_l2gate`, `ttt3r_random`
- 在 ScanNet 上取 10 个 scene，每个跑 200 帧，记录平均 per-frame 时间
- **实现**: 在 `inference_step` 入口/出口加 `torch.cuda.synchronize()` + `time.time()`，记录到 csv

### Exp S3: Hyperparameter Sensitivity

**SIASU τ (v1, 无意义)**: α ≡ 0.5，τ ∈ {0.5, 1.0, 2.0, 4.0} ATE: 0.280/0.281/0.286/0.283。无影响。

**Stability brake momentum_tau**: 从全量结果看 τ=1 (0.261) 明显优于 τ=2 (0.311)。需补充更多 τ 值（0.5, 1.5, 3.0）的 sensitivity 分析。

**Geo gate τ_g**: 从小规模 ablation 看 cutoff-insensitive（c2/c4/c8 均 ~-3.5%）。τ_g=2 最佳。

### Exp S4: Gate Visualization（v1 完成，暴露核心问题）

脚本: `analysis/s4_gate_visualization.py`，结果: `analysis_results/s4_gate_viz/summary.txt`。
- SIASU v1 alpha ≡ 0.5（零方差）
- ttt3r_mask 均值 0.30（非饱和）
- Geo gate 有真实方差（std=0.27）
- Effective mask 均值 0.079

### Cross-Attention Sparsity 验证（2026-03-26，Route C 前提检验）

脚本: `analysis/check_cross_attn_sparsity.py`

| 指标 | 值 | 含义 |
|------|-----|------|
| Normalized Entropy | 0.914 | 接近 uniform（1.0），非常 diffuse |
| % entropy < 0.8 | 0.0% | 没有任何 focused token |
| Effective patches | 419.5 / 737 (57%) | 每 token 有效关注 57% 的 image patch |
| Cosine similarity | 0.772 | token 间 attention pattern 高度相似 |

**结论**: Cross-attention 过于 diffuse，无法作为 pixel→token bridge。Route C1（通过 attention 投射 pixel gate 到 token space）不可行。

### Exp S5: Per-Scene Distribution → 升级为 Deep Analysis A3

已升级为 Paper 核心分析的一部分，见 Paper Narrative 中 Deep Analysis Plan A3。
- Scatter plot: x = random ATE, y = inv_t1 ATE，每点一个 scene
- 统计改善/退化 scene 数量 + worst case 分析
- 用已有 `eval_results/relpose/` 下的 per-scene error log 数据

## Next Steps

### 已完成
1. ~~Re-run Layer 2 SIASU ablation~~ Done (2026-03-23)
2. ~~Three-layer joint experiment~~ Done (2026-03-23). L23+ttt3r -7.5% best.
3. ~~Formal relpose eval ScanNet + TUM~~ Done (2026-03-24). ATE: ScanNet -68.1%, TUM -64.1%.
4. ~~Video Depth eval~~ Done (2026-03-24). KITTI -11.3%, Bonn -5.0%, Sintel -10.2%.
5. ~~3D Reconstruction eval~~ Done (2026-03-25). Acc -77.2%, Comp -54.2%.
6. ~~Exp S1 relpose~~ Done (2026-03-26). Naive baselines + stability brake 全量对比。
7. ~~Exp S4 gate visualization~~ Done (2026-03-26). SIASU alpha ≡ 0.5.
8. ~~Stability brake 实现 + 全量验证~~ Done (2026-03-27). TUM -20% vs random.
9. ~~A1/A2/A3 Deep Analysis~~ Done (2026-03-28). Gate std≈0.02, cos variance 不相关, per-scene ~50/50.
10. ~~Delta Clip / Attn Protect / Mem Novelty~~ Done (2026-03-28). 全部失败。
11. ~~Delta Orthogonalization TUM 验证~~ Done (2026-03-28). ATE 0.056, -15.4% vs constant.
12. ~~α_drift=0 消融~~ Done (2026-03-28). ATE 0.075, 确认 drift 含有用信号。

### 进行中
13. **[Running] ttt3r_ortho ScanNet 全量评测** — GPU0, 96 scenes

### 待办（按优先级）
14. **[P0] Ortho ScanNet 结果分析** — 等全量跑完，对比 stability brake 和 constant
15. **[P0] Ortho hyperparameter sensitivity** — α_novel ∈ {0.3, 0.5, 0.7}, α_drift ∈ {0, 0.05, 0.1, 0.2}, β ∈ {0.9, 0.95, 0.99}
16. **[P1] 最终方法 video depth + 7scenes 评测** — 用 ttt3r_ortho 作为最终方法
17. **[P1] Exp S2: Inference overhead** — wall-clock time + peak GPU memory
18. **[P1] 理论更新** — 补充 delta decomposition 的理论分析（drift energy bound, novel preservation）
19. **[P2] Ortho per-scene A3 分析** — scatter plot ortho vs constant, 验证 consistency