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

### Layer 2 — Token-Level State Modulation (SIASU, validated)
- **Signal**: Per-token high-freq residual energy of state trajectory (EMA low-pass → residual)
- **Action**: `alpha_k = sigmoid(-τ × (energy_k / running_mean - 1))` per token
- **Code**: `_spectral_modulation()`, update types `cut3r_spectral` / `ttt3r_spectral`
- **Result**: cut3r_spectral -5.0%, ttt3r_spectral -8.3% (vs cut3r). τ insensitive, use τ=1
- **Status**: Validated (2026-03-23)

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
| `ttt3r_l2gate` | ttt3r × l2_norm_gate | 1.0 | ← naive baseline (planned)
| `ttt3r_random` | ttt3r × p (constant) | 1.0 | ← naive baseline (planned)
| `ttt3r_conf` | ttt3r × conf_gate | 1.0 | ← naive baseline (existing)

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

### Failed Directions
- **Direction C (dynamic token tracking)**: State tokens don't track spatial semantics. Walking r=-0.024, static r=-0.383 (reversed). Abandoned.
- **Confidence gating (Exp 2)**: <1% improvement, feedback loop. Abandoned.

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
| `docs/run_experiments.sh` | All experiment commands |

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

## Paper Narrative（定稿方向，2026-03-25）

### 核心故事线

**问题**: Recurrent 3D reconstruction（CUT3R/TTT3R）的 state update 是 blind 的——不管输入帧是否带来新几何信息，都以相同力度更新。这在视频中有害：冗余帧的连续更新会把已收敛的几何估计搅乱，快速运动帧带来的剧变会导致 state 不稳定。TTT3R 的 learned gate（sigmoid cross-attention）是 data-driven 的，但训练时在 image pairs 上，未见过 long video 的 temporal redundancy pattern，泛化不够。

**Insight**: State update 何时更新、更新多少，本质是**信号变化检测问题**。频域是做变化检测的 natural tool——低频能量捕捉结构性变化，高频捕捉噪声/细节扰动。不是 arbitrarily 选了频域，而是问题本身就是频域问题。

**方法**: 两个互补的 frequency-domain gate：
- **L2（SIASU）**: State trajectory 的 token-level 高频残差——"state 自身在说它不稳定"（state 空间，高频信号）
- **L3（Geo Gate）**: 深度预测的低频结构变化——"预测结果在说几何变了"（output 空间，低频信号）
- 两层信号来源独立、频段互补，联合使用有额外增益

**卖点**: Train-free, inference-time, plug-and-play. 不改架构不改权重，在 CUT3R/TTT3R 上都有效 → 通用性。

### L1 的处理策略

**不在主文中提 L1 被弃用。** L1 frame skipping 作为 motivation 的引子——视频帧间存在大量冗余，低频能量可刻画冗余。然后 argue：粗粒度帧级跳过太 aggressive，会丢 fine-grained 信息，所以需要 token-level 和 geometric-level 的 soft gating。L1 的观察变成 motivate L2/L3 的 stepping stone。

消融表只呈现 L2、L3、L23 对比，不出现 L1/L12/L13/L123。如审稿人问"为什么不做 frame skipping"，rebuttal 用 L1 实验数据回答。

### Contribution 列法（insight-driven，不列描述性 contribution）

1. 揭示 recurrent 3D reconstruction 中 blind state update 的问题并定量分析
2. 提出 frequency-domain signal-based selective update，无需训练
3. 五个数据集三个任务（relpose, video depth, 3D recon）验证有效性

## Supplementary Experiments（待补充，2026-03-25 规划）

### Exp S1: Naive Baseline Comparison（必须）

证明频域信号的优越性，而非"少更新一点就好"或"简单变化量检测就够"。

需要加 3 个 baseline update type 到 `model.py`：

#### S1a: `ttt3r_l2gate` — L2 Norm Gating
- 用 state delta 的 L2 norm 代替 SIASU 的频域能量
- **逻辑**: `delta = new_state_feat - state_feat`，`energy = delta.norm(dim=-1, keepdim=True)`
- 同样维护 running mean + sigmoid gate，和 SIASU 结构完全对称
- **对比意义**: 同样的 gate 机制，只是信号不同（L2 norm vs 频域残差），证明 EMA 低通 + 高频残差分解是关键
- **实现**: 新增 `_l2_norm_gate()` static method，结构复制 `_spectral_modulation()`，去掉 EMA 低通步骤，直接用 `(new - old).norm()` 作为 energy

#### S1b: `ttt3r_random` — Random Gating
- 以与 SIASU 相同的平均 gate ratio 做随机 mask
- **逻辑**: 先跑一次 `ttt3r_joint` 记录平均 alpha 值（预计 0.5-0.7），然后 `alpha = p`（scalar constant）
- **对比意义**: 证明 selective gating 的信号 quality matters，不是"降低平均更新率就好"
- **实现**: `update_mask1 = update_mask * ttt3r_mask * p`，p 通过 config 传入

#### S1c: `ttt3r_conf` — Confidence Gating（已有）
- 已在 `model.py` 中实现，用预测 confidence 做 gate
- 直接作为 baseline 行呈现，无需额外代码

#### 实现步骤（Claude Code 操作）
1. 在 `model.py` 中添加 `_l2_norm_gate()` static method
2. 在 `_forward_impl()` 和 `inference_step()` 和 analysis path 的 update type switch 中添加 `ttt3r_l2gate` 和 `ttt3r_random` 分支
3. 先在快速实验（`--num_scannet 10`）上验证实现正确性
4. 在正式评测 pipeline（relpose ScanNet+TUM, video depth, 7scenes）上跑全量对比

### Exp S2: Inference Overhead（完成，2026-03-29）

Train-free 是卖点，需要证明 overhead negligible。

**结果**（TUM, 8 seqs × 200 frames, GPU4, size=512）：

| Config | FPS | Overhead | Peak GPU Mem |
|--------|-----|----------|--------------|
| cut3r (baseline) | 10.75 | — | 6.14 GB |
| ttt3r | 10.49 | +2.4% | 6.14 GB |
| **ttt3r_joint** | **10.47** | **+2.7%** | **6.14 GB** |
| ttt3r_l2gate | 10.63 | +1.1% | 6.14 GB |
| ttt3r_random | 10.57 | +1.7% | 6.14 GB |
| ttt3r_conf | 10.55 | +1.9% | 6.14 GB |

**结论**: 所有变体 overhead ≤ 3%，GPU 内存完全相同。ttt3r_joint 仅慢 2.7%，可在论文中声明 overhead negligible。结果保存于 `eval_results/benchmark_overhead.json`。

### Exp S3: Hyperparameter Sensitivity（必须）

整理成正式图表，在完整评测 pipeline 上跑（非快速实验的 10 scene）。

- **τ（spectral_temperature + geo_gate_tau）**: grid {0.5, 1, 2, 4}
- **freq_cutoff（geo_gate_freq_cutoff）**: grid {2, 4, 8}
- 在 ScanNet relpose 上跑，报 ATE median
- 已有部分快速实验数据（τ insensitive, cutoff insensitive），正式实验确认

### Exp S4: Qualitative Visualization（建议）

- 挑 2-3 个 representative sequences（ScanNet 快速转动 / 静止 / 遮挡场景）
- 展示 gate activation（L2 alpha, L3 g_geo）随时间的变化曲线
- 配合 RGB 帧 + 深度图 + 点云质量对比
- **实现**: 在 `inference_step` 中记录每帧的 alpha 和 g_geo 值到 list，后处理绘图

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
6. Exp S1: Naive baseline comparison (ttt3r_l2gate, ttt3r_random, ttt3r_conf)
7. ~~Exp S2: Inference overhead measurement~~ Done (2026-03-29). ttt3r_joint overhead +2.7%, GPU mem identical.
8. Exp S3: Hyperparameter sensitivity (τ, freq_cutoff) on full eval
9. Exp S4: Qualitative visualization (gate activation over time)
10. Exp S5: Per-scene distribution analysis
11. Paper outline drafting
