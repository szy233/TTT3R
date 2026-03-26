# TTT3R — Frequency-Guided State & Memory Update Framework

## Project Goal
NeurIPS submission. Train-free, inference-time frequency-domain framework for selective state/memory updates in recurrent 3D reconstruction (CUT3R/TTT3R).

## Architecture Overview

The model (`src/dust3r/model.py`, class `ARCroco3DStereo`) processes video frames recurrently:
1. Encode frame → `feat_i`
2. `_recurrent_rollout(state_feat, feat_i)` → `new_state_feat`, `dec`
3. `pose_retriever.update_mem(mem, feat, pose)` → `new_mem`
4. `_downstream_head(dec)` → `res` (pts3d, conf)
5. State update: `state_feat = new * mask1 + old * (1-mask1)`
6. Memory update: `mem = new_mem * mask2 + mem * (1-mask2)`

`mask1` and `mask2` are where our frequency-domain gates are applied.

## Three-Layer Frequency Framework

### Layer 1 — Frame Filtering (validated)
- **Signal**: `LFE(FFT2(RGB_diff))` — low-freq energy of inter-frame RGB difference
- **Action**: Skip frames where LFE < threshold × EMA(LFE)
- **Result**: Skip 35% frames, TTT3R depth -3.1% on ScanNet
- **Code**: `compute_frame_spectral_change()`, `filter_views_by_spectral_change()`

### Layer 2 — Token-Level State Modulation (SIASU)
- **Signal**: Per-token high-freq residual energy of state trajectory (EMA low-pass → residual)
- **Code**: `_spectral_modulation()`, update types `cut3r_spectral` / `ttt3r_spectral`
- **Status**: ⚠️ **v1 退化为常数** (2026-03-26)
  - v1 公式: `alpha = sigmoid(-τ × (energy / running_energy - 1))`，EMA γ=0.95 紧密追踪 energy → ratio ≈ 1.0 → alpha ≡ 0.5（S4 实验确认：token/frame 方差均为 0）
  - 之前 "L2+ttt3r -8.2%" 的提升**全部来自乘常数 0.5**，非频域信号
  - **v2 修复 (testing)**: 改用 cross-token ranking，`percentile = energy / max(energy)`，`alpha = sigmoid(-τ × (percentile - 0.5))`。保证 token 间有差异（by construction）
  - v2 正在 ScanNet relpose 全量评测中（`eval_results/relpose/scannet_s3_1000/ttt3r_joint_siasu_v2/`）

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

### S1 Naive Baseline Comparison (2026-03-26)

**Relpose ATE (Average) — ScanNet**

| Config | ATE ↓ | vs cut3r | vs ttt3r |
|--------|-------|----------|----------|
| cut3r (baseline) | 0.817 | — | — |
| ttt3r | 0.406 | -50.3% | — |
| **ttt3r_joint** | **0.283** | -65.4% | -30.3% |
| ttt3r_l2gate | 0.276 | -66.3% | -32.1% |
| ttt3r_random (p=0.5) | 0.280 | -65.8% | -31.1% |
| ttt3r_conf | 0.298 | -63.5% | -26.6% |

**Relpose ATE (Average) — TUM**

| Config | ATE ↓ | vs cut3r | vs ttt3r |
|--------|-------|----------|----------|
| cut3r (baseline) | 0.166 | — | — |
| ttt3r | 0.103 | -38.1% | — |
| **ttt3r_joint** | **0.069** | -58.6% | -33.5% |
| ttt3r_l2gate | 0.077 | -53.5% | -25.2% |
| ttt3r_random (p=0.5) | 0.079 | -52.4% | -23.3% |
| ttt3r_conf | 0.073 | -56.1% | -29.1% |

**⚠️ 关键发现**: naive baselines（l2gate, random, conf）在 relpose 上与 ttt3r_joint 效果相当。原因：SIASU v1 alpha ≡ 0.5，所以 ttt3r_joint 本质就是 `ttt3r × 0.5 × g_geo`，而 naive baselines 也是 `ttt3r × ~0.5`。ttt3r_joint 在 TUM 上仍有优势（0.069 vs 0.077-0.079），可能来自 geo gate 的贡献。

**S1 video depth + 7scenes 尚未完成**（`run_baseline_eval.sh` 中 video depth 数据集名错误 `kitti_1000` → 应为 `kitti_s1_500`，需修复后重跑）。

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
- **SIASU v1 (energy/running_energy normalization)**: EMA 紧密追踪 → ratio ≈ 1.0 → alpha ≡ 0.5. 退化为常数乘子，无 selective gating 能力。已替换为 v2 (cross-token ranking)。
- **Route C1 (cross-attention bridge)**: Decoder cross-attention 太 diffuse（normalized entropy 0.914, cosine sim 0.772），无法将 pixel-space gate 有效传递到 token space。Token gate 退化为 scalar mean(pixel_gate)。
- **原始 momentum gate (non-inverted)**: cos~0.74 → gate~0.80 → 几乎不 dampening → 比常数 0.5 差。SGD momentum 直觉在 over-update 场景下有害。

### Inverted Momentum Gate / Stability Brake (2026-03-26)

**Bug 修复后新方向**: 反转 momentum gate — `sigmoid(-τ × cos)` 代替 `sigmoid(τ × cos)`

**早期结果 (16 scenes ScanNet + 8 TUM)**:

| Config | ScanNet ATE ↓ | TUM ATE ↓ | TUM vs random |
|--------|--------------|-----------|---------------|
| ttt3r_random (×0.5) | 0.265 | 0.073 | — |
| ttt3r_momentum_inv_t2 (τ=2) | 0.267 | 0.062 | -15.1% |
| **ttt3r_momentum_inv_t1 (τ=1)** | **0.263** | **0.057** | **-21.9%** |
| ttt3r_joint_fixed | 0.307* | 0.072 | — |
| ttt3r_momentum_v2 (non-inverted) | 0.370* | 0.091 | +24.7% |

**关键发现**:
- inv_t1 在 TUM 上大幅优于 random（-22%），证明自适应 dampening 有独立价值
- ScanNet 上 inv_t1 ≈ random，符合理论预测（室内静态，cos 方差小 → 退化为常数）
- 全量 ScanNet 评测进行中

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

对比三个配置：`cut3r`（baseline）, `ttt3r`, `ttt3r_joint`（L23+ttt3r，最终方法）。

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

## Paper Narrative（更新方向，2026-03-26）

### 核心故事线

**问题**: Recurrent 3D reconstruction（CUT3R/TTT3R）的 state update 存在 over-update——不管输入帧是否带来新几何信息，都以相同力度更新。任何 dampening（×0.5）都显著改善，说明 systematic over-update 是核心问题。TTT3R 的 learned gate（sigmoid cross-attention）训练在 image pairs 上，未见过 long video 的 temporal pattern，泛化不够。

**Insight**: State update 何时更新、更新多少，可形式化为在线优化问题。常数 dampening 面临 stability-reactivity tradeoff（静态场景需小 α 防过冲，动态场景需大 α 跟踪变化）。自适应 dampening 可通过 state trajectory 的收敛指标（连续更新的 cosine alignment）同时优化两端。

**方法**: 两个互补的 train-free gate：
- **Stability Brake (state space)**: `α_t = σ(-τ·cos(δ_t, δ_{t-1}))` — 检测 state 轨迹收敛（cos 高→制动），突变时放行（cos 低→更新）
- **Geo Gate (output space)**: `g_t = σ(-τ_g·LFE(FFT2(Δlog_depth)))` — 检测深度预测不一致时抑制更新
- 两者信号独立（state space vs output space），失效模式互补

**理论支撑** (`docs/theory_section.tex`):
1. Over-update bound: 无 dampening 时误差 O(k²) 增长
2. Regret bound: 自适应 dampening 严格优于常数（混合运动序列）
3. Optimal τ: 与运动多样性相关，退化到常数的条件可解释

**卖点**: Train-free, inference-time, plug-and-play + 理论保证。

### Contribution

1. 揭示 recurrent 3D reconstruction 中 over-update 问题，证明误差 O(k²) 增长 bound
2. 提出 adaptive dampening (stability brake + geo gate)，证明 regret 严格优于常数
3. 五个数据集三个任务验证有效性，TUM relpose -22% vs constant dampening

## Supplementary Experiments（2026-03-26 更新）

### Exp S1: Naive Baseline Comparison（relpose 完成，video depth/7scenes 跑中）

已实现 `ttt3r_l2gate`, `ttt3r_random`, `ttt3r_conf` 三个 naive baseline。Relpose (ScanNet + TUM) 完成，结果见上方。

**已修复**: `run_baseline_eval.sh` 数据集名（`kitti_1000` → `kitti_s1_500` 等）。
**当前状态**: Video depth + 7scenes 双 GPU 并行跑中：
- GPU0: KITTI(3) + Sintel(3) + 7scenes/l2gate（等 S3 st4.0 完成后自动启动，log: `eval/s1_gpu0.log`）
- GPU1: Bonn(3) + 7scenes/random + conf（等 SIASU v2 完成后自动启动，log: `eval/s1_gpu1.log`）
- **🔥 关键待验证**: ttt3r_random 在 7scenes 上是否和 ttt3r_joint 效果相当 → 决定 geo gate 是否有价值

### Exp S2: Inference Overhead（必须）

Train-free 是卖点，需要证明 overhead negligible。

- 每个配置跑 wall-clock time 和 peak GPU memory
- 配置: `cut3r`, `ttt3r`, `ttt3r_joint`, `ttt3r_l2gate`, `ttt3r_random`
- 在 ScanNet 上取 10 个 scene，每个跑 200 帧，记录平均 per-frame 时间
- **实现**: 在 `inference_step` 入口/出口加 `torch.cuda.synchronize()` + `time.time()`，记录到 csv

### Exp S3: Hyperparameter Sensitivity（v1 结果无意义）

v1 SIASU alpha ≡ 0.5，τ 无影响。已跑 τ ∈ {0.5, 1.0, 2.0, 4.0}，ATE: 0.280/0.281/0.286/(running)。
结果: `eval_results/relpose/scannet_s3_1000/ttt3r_joint_st*`。
SIASU v2 的 τ sensitivity 视 v2 是否有效决定是否重跑。

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

### Exp S5: Per-Scene Distribution（建议）

- Scatter plot: x=cut3r ATE, y=ttt3r_joint ATE, 每点一个 scene
- 或 box plot 展示改善分布的 consistency
- 用已有 `eval_results/relpose/` 下的 per-scene error log 数据

## Next Steps
1. ~~Re-run Layer 2 SIASU ablation (warm-start fixed)~~ Done (2026-03-23)
2. ~~Three-layer joint experiment (Layer 1 + 2 + 3)~~ Done (2026-03-23). L23+ttt3r -7.5% best; L1 conflicts.
3. ~~Formal relpose eval on ScanNet + TUM~~ Done (2026-03-24). ATE: ScanNet -68.1%, TUM -64.1%.
4. ~~Video Depth eval~~ Done (2026-03-24). Abs Rel: KITTI -11.3%, Bonn -5.0%, Sintel -10.2%.
5. ~~3D Reconstruction eval (需下载 7scenes)~~ Done (2026-03-25). Acc -77.2%, Comp -54.2% (ttt3r_joint).
6. ~~Exp S1 relpose~~ Done (2026-03-26). Naive baselines 与 joint 效果相当（ScanNet ATE: joint 0.283, l2gate 0.276, random 0.280）。
7. ~~Exp S4 gate visualization~~ Done (2026-03-26). SIASU v1 alpha ≡ 0.5，geo gate 有真实方差。
8. ~~Exp S3 τ sensitivity (v1)~~ Done (2026-03-26). τ 无影响（alpha ≡ 0.5）。
9. ~~Cross-attention sparsity check~~ Done (2026-03-26). Diffuse（entropy 0.914），Route C1 不可行。
10. ~~Gate state 每帧重置 bug~~ Fixed (2026-03-26). `if reset_mask.any():` 三处修复。
11. ~~反转 momentum gate 实现~~ Done (2026-03-26). `sigmoid(-τ × cos)`, TUM -22% vs random.
12. ~~理论推导~~ Done (2026-03-26). `docs/theory_section.tex`, 三个命题 + 证明。
13. **[Running] Stability brake 全量 ScanNet** — GPU0: inv_t2, GPU1: inv_t1
14. **[Running] 远程服务器 (wh@10.160.4.17)** — GPU4: TUM + ScanNet inv/joint_fixed/l2gate_fixed
15. **[Todo] ttt3r_brake_geo** — stability brake + geo gate 联合（最终方法）
16. **[Todo] Exp S2: Inference overhead**
17. **[Todo] Exp S5: Per-scene distribution**
18. **🔥 决策点**: 全量 ScanNet inv_t1 结果 → 确认 adaptive dampening 在大数据集上的优势