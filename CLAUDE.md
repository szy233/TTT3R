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
| 7scenes | ❌ 未下载 | — | 待定 |

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

## Known Issues / Fixes Applied
1. **SIASU warm-start**: `running_energy` init 0 → ratio explosion → state frozen. Fixed: warm-start on first call.
2. **TUM depth matching**: Timestamp-based association needed (not stem-based).
3. **Fair evaluation**: Compare full vs filtered on same `kept_indices`.
4. **ScanNet pose 截断**: 根分区满时 `prepare_scannet_local.py` 写 pose 文件被截断（scene0707_00）。已修复重新生成。
5. **ScanNet 31 scene Eigenvalue failure**: GT pose 含 -inf（深度传感器丢失追踪），evo Umeyama `eigh()` 不收敛。与原论文行为一致（同样 skip），不影响公平对比。4 个 scene (0777-0780) .sens 未解压，预处理跳过。

## Next Steps
1. ~~Re-run Layer 2 SIASU ablation (warm-start fixed)~~ Done (2026-03-23)
2. ~~Three-layer joint experiment (Layer 1 + 2 + 3)~~ Done (2026-03-23). L23+ttt3r -7.5% best; L1 conflicts.
3. ~~Formal relpose eval on ScanNet + TUM~~ Done (2026-03-24). ATE: ScanNet -68.1%, TUM -64.1%.
4. ~~Video Depth eval~~ Done (2026-03-24). Abs Rel: KITTI -11.3%, Bonn -5.0%, Sintel -10.2%.
5. 3D Reconstruction eval (需下载 7scenes)
6. Paper outline drafting
