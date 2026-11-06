# Analysis Scripts

## 论文核心分析 (A1-A7)

论文中直接引用的分析实验，产出支撑论文 claims 的数据和图表。

| 脚本 | 分析编号 | 功能 | 输出���录 |
|------|---------|------|---------|
| `a1a2_gate_dynamics.py` | A1, A2 | **A1**: 逐帧可视化 cos(δ_t, δ_{t-1})、gate α_t、GT camera motion 的时序曲线，展示 TTT3R scalar gate 退化为常数（σ≈0.02）。**A2**: 收集所有 scene 的 Var(cos) 与 brake 改善幅度做 scatter，证明 cosine variance 与改善无相关 (r=-0.13, p=0.63) | `analysis_results/a1a2_dynamics/` |
| `a1_outdoor.py` | A1 补充 | A1 在 KITTI + Sintel 上的扩展。从 .cam 文件读 Sintel GT pose，从 KITTI 数据读 GT | 同上 |
| `a3_per_scene_distribution.py` | A3 | Per-scene ATE 分布对比：brake vs constant。scatter (x=random ATE, y=brake ATE) + boxplot。证明 brake 在约 50% 场景改善 | `analysis_results/a3_per_scene/` |
| `a4_delta_direction.py` | A4 | Delta 方向分析：跑所有 ScanNet + TUM scene，统计 cos(δ_t, δ_{t-1}) 和 drift energy (cos²) 的 per-scene 分布。核心发现：TUM drift energy 0.40 vs ScanNet 0.60 | `analysis_results/a4_delta_direction/` |
| `taum_gate_stats.py` | A5 | 用 `cut3r_taum_log` 模式模拟 TTSA3R 的 TAUM gate，记录 temporal/spatial/final gate 的统计量。证明 TAUM gate σ_time=0.006，退化为常数 ~0.355 | `analysis_results/taum_gate_stats/` |
| `token_gate_variance.py` | A1 补充 | 检查 brake gate 的 per-token 方差：即使 frame-level 均值近似常数 (~0.33)，token-level 是否有有意义的空间选择性 | `analysis_results/` |
| `viz_scatter_drift_ortho.py` | A7 | Per-scene scatter: drift energy vs ortho improvement (ScanNet 90f)。核心图表：证明 ortho 与 drift energy 显著正相关 (r=0.248, p=0.018)，多方法对比 (ortho/brake/random) | `analysis_results/scatter_drift_ortho/` |

## 论文可视化

生成论文 figure 的脚本。

| 脚本 | 功能 | 输出目录 |
|------|------|---------|
| `viz_depth_qualitative.py` | Depth 定性对比图：RGB \| GT \| cut3r \| ttt3r \| brake \| ortho，附带 error map。用于 Bonn balloon2 等序列 | `analysis_results/depth_qualitative/` |
| `viz_traj_comparison.py` | 轨迹对比可视化：多方法 Sim(3) aligned BEV 投影。暗底风格，匹配 TTT3R 论文 Figure 16 | `analysis_results/traj_comparison/` |
| `s4_gate_visualization.py` | Gate 激活时序图：记录 ttt3r_joint 下各 gate 分量 (ttt3r_mask, SIASU alpha, geo g_geo) 的逐帧变化 | `analysis_results/s4_gate_viz/` |

## 早期探索实验（已放弃的方向）

这些脚本产出了探索阶段的数据，对应已放弃的 update type（spectral, geogate, memgate, joint 等）。代码保留作为研究记录，不再用于论文。

| 脚本 | 功能 |
|------|------|
| `spectral_analysis.py` | 将 state token 轨迹做频域分解（FFT），分析各频带能量与重建误差的相关性 |
| `spectral_ablation.py` | spectral modulation (SIASU) 消融：对比 cut3r_spectral / ttt3r_spectral 在不同 temperature τ 下的 depth error |
| `batch_spectral.py` | 在多 scene 上批量跑 spectral analysis，聚合 per-scene 频域特征与 depth error 的相关性 |
| `geogate_ablation.py` | geometric consistency gate 消融：对比 cut3r/ttt3r/cut3r_geogate/ttt3r_geogate，扫描 geo_gate_tau |
| `memgate_ablation.py` | memory gate 消融：对比 spectral-change gated memory update 在不同超参下的效果 |
| `joint_ablation.py` | 三层联合消融（TTT3R × SIASU × GeoGate）：并行评估所有 layer 组合的独立与联合贡献 |
| `conf_gated_ablation.py` | confidence-gated state update 消融：对比 cut3r / ttt3r / ttt3r_conf 的 depth error 和 confidence calibration |

## 早期探索分析（已放弃的方向）

| 脚本 | 功能 |
|------|------|
| `state_freq_analysis.py` | State token 时域频率可视化：用 per-token temporal variance 做 heatmap，检验高频 token 是否对应动态区域 |
| `freq_error_analysis.py` | Token 频率 vs 重建误差相关性分析：scatter、时序相关、逐帧可视化 |
| `frame_level_analysis.py` | 逐帧 state change / confidence / TTT3R mask 与 depth error 的相关性分析 |
| `batch_frame_level.py` | 在多 ScanNet scene 上批量跑 frame_level_analysis，聚合 per-scene 相关系数 |
| `frame_novelty_analysis.py` | 帧间 spectral change 与 state oscillation 的关系：验证跳过低 spectral_change 帧是否减少 state 震荡 |
| `batch_frame_novelty.py` | 在多 scene 上批量跑 frame_novelty_analysis |
| `dynamic_token_analysis.py` | 动态 token 分析：EMA → 高频残差能量 ��� 通过 cross-attention 投射到图像空间 → 验证是否对应动态物体 |
| `metric_comparison.py` | 帧间变化指标对比：spectral_change / l2_change / high_freq_change / mid_freq_change vs state oscillation 相关性 |
| `check_cross_attn_sparsity.py` | 快速检查 decoder cross-attention 的稀疏性 vs 弥散性 |
| `check_gradient_alignment.py` | 快速检查 consecutive state delta 的 cosine similarity 是否有有意义的变化 |

## 运行说明

所有脚本需要在项目根目录下运行，典型命令格式：

```bash
conda activate ttt3r
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python analysis/<script>.py [args]
```

论文核心分析 (A1-A7) 的依赖关系：
- A4 需要先跑完产出 `a4_summary.txt`，A7 scatter 脚本依赖该文件
- A1/A2 可独立运行
- A3 依赖 eval_results 中已有的 per-scene ATE 数据
- A5 可独立运行
