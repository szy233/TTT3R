# H200 Server Runlog (2026-03-29)

## 1) 环境与硬件

- 服务器: `NVIDIA H200 x1` (`143771 MiB`)
- CPU/内存: `8 vCPU / 141 GB RAM`
- 仓库分支: `zjc` (commit `c7ba452`)
- Python 环境: `python3.10 + .venv`
- 关键运行参数:
  - `NUM_PROCESSES=1`
  - `SIZE=512`
  - `AMP_DTYPE=bf16`
  - `TF32=1`
  - `CUDNN_BENCHMARK=1`
  - `INFERENCE_MODE=1`

## 2) 数据准备

- 已下载并校验模型权重:
  - `src/cut3r_512_dpt_4_64.pth` (`~3.0G`)
- 已下载并解压:
  - `nuScenes v1.0-mini` 到 `/root/datasets/nuscenes`
- Waymo 数据状态:
  - 当前服务器未检测到任何 `*.tfrecord`
  - 公开直链访问返回 `403`，说明需账号授权后的 Waymo 数据源

## 3) nuScenes 实验（已完成）

执行脚本:

```bash
bash scripts/server/run_nuscenes_relpose_pipeline.sh
```

数据转换输出:

- `data/nuscenes_relpose/` (10 个序列，单序列约 39~41 帧)

评测输出:

- `eval_results/relpose/nuscenes_relpose/summary.csv`
- `eval_results/relpose/nuscenes_relpose/per_sequence_results.csv`
- `eval_results/relpose/nuscenes_relpose/summary.md`
- `eval_results/relpose/nuscenes_pipeline.log`

### 3.1 结果摘要 (summary.csv)

| model | avg_ate | avg_rpe_trans | avg_rpe_rot |
|---|---:|---:|---:|
| cut3r | 2.57783 | 1.25918 | 0.90599 |
| ttt3r | 10.04719 | 4.88576 | 1.31527 |
| ttt3r_momentum_inv_t1 | 17.99197 | 7.99655 | 10.64272 |
| ttt3r_momentum_inv_t1_drift0 | 17.99197 | 7.99655 | 10.64272 |

备注:

- 在本次 nuScenes mini 子集上，`cut3r` 明显优于三组 TTT3R 配置。
- `ttt3r_momentum_inv_t1` 与 `ttt3r_momentum_inv_t1_drift0` 指标完全一致，和之前在 KITTI 观察到的现象一致，建议后续继续排查该分支的 gate 生效路径。

## 4) Waymo 状态与下一步

当前阻塞不是代码错误，而是 **数据访问权限**:

1. Waymo 数据未挂载到服务器（当前 `tfrecord` 数量为 0）。
2. 公开 URL 返回 403，说明必须使用有权限的账号/数据盘。

拿到可用 Waymo 数据后可直接运行:

```bash
export WAYMO_TFRECORD_GLOB="/path/to/waymo/*.tfrecord"
export NUM_PROCESSES=1
export SIZE=512
export AMP_DTYPE=bf16
export TF32=1
export CUDNN_BENCHMARK=1
export INFERENCE_MODE=1
bash scripts/server/run_waymo_relpose_pipeline.sh
```

