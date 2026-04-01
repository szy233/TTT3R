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

## CLI 参数

| Paper Symbol | CLI Arg | Default |
|-------------|---------|---------|
| α (constant) | `--alpha` | 0.5 |
| α⊥ | `--alpha_perp` | 0.5 |
| α∥ | `--alpha_parallel` | 0.05 |
| β_ema | `--beta_ema` | 0.95 |
| γ | `--gamma` | 0.0 |
| τ (brake) | `--brake_tau` | 2.0 |
| auto_gamma | `--auto_gamma` | `""` |
| auto_gamma_warmup | `--auto_gamma_warmup` | 30 |
| auto_gamma_max | `--auto_gamma_max` | 3.0 |
| auto_gamma_k | `--auto_gamma_k` | 10.0 |
| auto_gamma_lo/hi | `--auto_gamma_lo/hi` | 0.3 / 0.6 |

Old CLI args (`--ortho_alpha_novel`, `--ortho_beta`, etc.) still work as hidden aliases.

**Auto-gamma modes**: `warmup_linear`, `warmup_threshold` (sequence-level), `steep_sigmoid`, `steep_clamp` (per-frame)

## Eval

```bash
# Unified eval script
bash eval/run_ddd3r_eval.sh <GPU> <DATASET> <METHOD>
# e.g.: bash eval/run_ddd3r_eval.sh 0 tum_s1_1000 ddd3r
#        bash eval/run_ddd3r_eval.sh 0 scannet_s3_1000 ddd3r_auto_warmup_linear

# Datasets: tum_s1_1000, tum_s1_90, scannet_s3_1000, scannet_s3_90, sintel, kitti_odom
#           kitti, bonn, sintel_depth (video depth), 7scenes (3D recon)
# Methods:  cut3r, ttt3r, ddd3r_constant, ddd3r_constant_p{N}, ddd3r_brake, ddd3r, ddd3r_g{N}
#           ddd3r_auto_warmup_linear, ddd3r_auto_warmup_threshold
#           ddd3r_auto_steep_sigmoid, ddd3r_auto_steep_sigmoid_k20
#           ddd3r_auto_steep_clamp, ddd3r_auto_steep_clamp_tight

# Auto-gamma parallel eval
bash eval/run_auto_gamma_eval.sh 0,1

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
| `analysis/` | A1-A7 分析脚本 + 可视化 |

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
- 🔄 ScanNet scaling curve (200f/500f × 6 methods)
- 🔄 Auto-gamma experiment (6 methods × 3 datasets)
- ⬜ Analysis section writing (M1→M2→M3)
- ⬜ Experiments section (5 datasets × 3 tasks + spectrum ablation)
- ⬜ Intro + related work
