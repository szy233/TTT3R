# DDD3R — Directional Decomposition and Dampening for Recurrent 3D Reconstruction

NeurIPS submission. 诊断 recurrent 3D reconstruction 中 state update 调节的系统性失效，提出 DDD3R 作为统一解决方案。详细叙事见 `docs/paper_narrative.md`，实验结果见 `docs/experiment_results.md`。

## Architecture

Model: `src/dust3r/model.py`, class `ARCroco3DStereo`. State update 在 `mask1` 处应用：
```
state_feat = new * mask1 + old * (1-mask1)
```

**统一 update rule：** `S_t = S_{t-1} + β_t (α⊥ · δ⊥ + α∥ · δ∥)`

三个 stage: Decompose (EMA drift direction → 投影分解) → Reweight (α⊥ > α∥) → Gate (β_t spatial mask)

## Update Types

| `model_update_type` | 行为 | Paper Role |
|---------------------|------|------------|
| `cut3r` | mask1=1.0 | baseline |
| `ttt3r` | sigmoid(cross_attn) | existing method |
| `ddd3r_constant` | ttt3r × α | α⊥=α∥=α special case |
| `ddd3r_brake` | ttt3r × stability brake | baseline (一阶近似) |
| `ddd3r` | directional decomposition | DDD3R with α⊥>α∥, γ≥0 |
| `ddd3r` + `--auto_gamma entropy` | attention entropy adaptive | α_∥^(t) = h̄_t·α⊥ + (1-h̄_t)·α∥ |
| `ddd3r` + `--auto_gamma drift_energy` | drift energy adaptive | α_∥^(t) = e_t·α⊥ + (1-e_t)·α∥, directly measures drift |

## CLI 参数

| Paper Symbol | CLI Arg | Default |
|-------------|---------|---------|
| α (constant) | `--alpha` | 0.5 |
| α⊥ | `--alpha_perp` | 0.5 |
| α∥ | `--alpha_parallel` | 0.05 |
| β_ema | `--beta_ema` | 0.95 |
| γ | `--gamma` | 0.0 |
| τ (brake) | `--brake_tau` | 1.0 |
| auto_gamma | `--auto_gamma` | `""` |
| β_entropy | `--entropy_ema_beta` | 0.95 |
| auto_gamma_warmup | `--auto_gamma_warmup` | 30 |
| auto_gamma_max | `--auto_gamma_max` | 3.0 |
| auto_gamma_k | `--auto_gamma_k` | 10.0 |
| auto_gamma_lo/hi | `--auto_gamma_lo/hi` | 0.3 / 0.6 |

Old CLI args (`--ortho_alpha_novel`, `--ortho_beta`, etc.) still work as hidden aliases.

**Auto-gamma modes**: `warmup_linear`, `warmup_threshold` (sequence-level), `steep_sigmoid`, `steep_clamp` (per-frame), `entropy` (attention entropy adaptive, zero-cost), `drift_energy`, `local_de`, `local_de_raw`, `local_de_raw_p2`, `local_de_fmean_sig`, `local_de_fmean`, `local_de_token`, `local_de_token_sig`, `drift_growth`, `proj_frac`, `momentum_R`

## Eval

```bash
# Unified eval script
bash eval/run_ddd3r_eval.sh <GPU> <DATASET> <METHOD>
# e.g.: bash eval/run_ddd3r_eval.sh 0 tum_s1_1000 ddd3r
#        bash eval/run_ddd3r_eval.sh 0 scannet_s3_1000 ddd3r_auto_warmup_linear

# Datasets (TTT3R standard frame configs):
#   Relpose ScanNet: scannet_s3_{50,90,100,150,200,...,1000} (21 points)
#   Relpose TUM:     tum_s1_{50,100,150,200,300,...,1000} (12 points)
#   Video Depth KITTI: kitti_s1_{50,100,150,...,500} (10 points)
#   Video Depth Bonn:  bonn_s1_{50,100,150,...,500} (10 points)
#   Fixed: sintel (relpose), sintel_depth (video depth), kitti_odom, 7scenes (3D recon)
# Methods:  cut3r, ttt3r, ddd3r_constant, ddd3r_constant_p{N}, ddd3r_brake, ddd3r, ddd3r_g{N}
#           ddd3r_entropy, ddd3r_de, ddd3r_local_de, ddd3r_local_de_raw_p2, ddd3r_fmean_sig
#           ddd3r_drift_growth, ddd3r_proj_frac, ddd3r_momentum, ddd3r_boost{N}, ddd3r_a{N}

# Relpose scaling curve (112 jobs, 2-GPU parallel)
bash eval/run_scaling_curve.sh 0 1
# Video depth scaling curve (82 jobs, chained after relpose)
bash eval/run_vdepth_scaling_curve.sh 0 1

# Portable: DDD3R_PYTHON=/path/to/python bash eval/run_ddd3r_eval.sh ...
```

**Protocols**: Relpose=Sim(3) ATE RMSE, Video Depth=scale&shift abs_rel, 3D Recon=Acc/Comp/NC

## Key Files

| File | Purpose |
|------|---------|
| `src/dust3r/model.py` | 所有 update types, `_delta_ortho_update` |
| `docs/paper_narrative.md` | Paper 叙事 + 路线决策 |
| `docs/experiment_results.md` | 所有实验结果 + Analysis |
| `docs/research_progress.md` | 完整研究日志 |
| `eval/run_ddd3r_eval.sh` | 统一评测脚本 |
| `eval/run_auto_gamma_eval.sh` | Auto-gamma 并行评测 |
| `analysis/` | A1-A12 分析脚本 + 可视化 |
| `eval/relpose/kitti_odo_full_report.md` | KITTI full 14methods×11seqs 完整报告（zjc 分支） |

## Paths
- Model weights: `model/cut3r_512_dpt_4_64.pth`
- Datasets: `/mnt/sda/szy/research/dataset/` (ScanNet, TUM)
- ScanNet: 100 test → 96 preprocessed → 90 valid (90f) / 65-66 valid (1000f)
- TUM: 8 sequences, all valid

## GhostGPU 协作

服务器部署了 GhostGPU 显存占位守护进程。**实验只需 ~6G 显存，目标卡降到 12% 预留 ~10G 即可，剩余显存继续占位：**

```bash
# 单卡实验：目标卡降到12%，留~10G给实验，其余继续占；跑完恢复60%
blend 0 12 && bash eval/run_ddd3r_eval.sh 0 tum_s1_1000 ddd3r ; blend 0 60

# 多个实验串行（同一张卡）
blend 0 12 && bash eval/run_ddd3r_eval.sh 0 tum_s1_1000 ddd3r && bash eval/run_ddd3r_eval.sh 0 scannet_s3_1000 ddd3r ; blend 0 60

# 双卡并行实验
blend 0 12 && blend 1 12 && bash eval/run_auto_gamma_eval.sh 0,1 ; blend 0 60 && blend 1 60
```

**规则：**
- 实验前目标卡降到 12%：`blend <GPU_ID> 12`（非目标卡不动）
- 实验结束后恢复：`blend <GPU_ID> 60`
- 不要用 `coffee`（会释放全部卡全部显存）
- GPU ID 从 eval 命令的第一个参数读取

## Known Issues
1. Gate state reset: `view["reset"]` returns `tensor([False])` not None → use `reset_mask.any()`. Fixed.
2. ScanNet scene skip: GT contains -inf, evo eigh fails. Consistent across configs.

## Next Steps
- ✅ Attention entropy adaptive (ddd3r_entropy: TUM 0.070, ScanNet 0.294)
- ✅ All auto-gamma variants (warmup, steep, entropy, drift_energy, local_de, per-token)
- ✅ KITTI Odom full eval (14 methods × 11 seqs) — ddd3r_g1 ATE best (-22.7%)
- ✅ Paper all sections initial draft
- ✅ All adaptive methods exhausted (drift_growth, proj_frac, momentum, fmean_sig, boost) — Pareto frontier confirmed
- ✅ α∅ ablation (a10-a25): TUM robust (0.055-0.056), ScanNet monotonically improves (a20=0.367, a25=0.344)
- ✅ 论文方向 Narrative C：brake 主方法 + scaling curve + ortho analysis
- ✅ Relpose scaling curve（132 jobs，130 完成，缺 tum_s1_150/brake + scannet_s3_500/brake 异常）
  - ScanNet 21 points × 4 methods + TUM 12 points × 5 methods (含 ddd3r)
- ✅ Overnight ablation (α∅ ablation ScanNet a20/a25 完成)
- ✅ Drift direction confidence gate (drift_conf/token/fallback) — ScanNet 上失败，drift dir 也稳定
- ✅ Ortho + Brake 叠加 (ortho_brake) — 严重退化 (TUM 0.107)，信号空间不兼容
- ✅ 自适应方案最终确认穷尽：所有 online 信号维度已探索，Pareto frontier 无法突破
- ✅ drift_conf_token + drift_conf_fallback ScanNet — 均失败（≈ortho）
- ✅ ortho_brake ScanNet — 严重退化 (0.589)，OOM killed
- ✅ Video depth scaling curve — KITTI/Bonn 10pts×5methods + Sintel 全部完成
  - Bonn: brake 全长度最优 (~8% vs ttt3r)
  - KITTI: 300f 交叉点，短序列 brake > ortho，长序列 ortho > brake
- ⬜ 根据最终 scaling curve 数据更新 paper method/experiments sections
- ⬜ .bib file (cite keys are placeholders)
- ⬜ Figures: scaling curve (主图), method diagram, ablation table, qualitative vis
- ⬜ Appendix (per-scene tables, adaptive negative results, hyperparameter sweeps)
