# TTT3R 实验结果总结（zjc 分支）

> 本文档只保留实验结果，不展开方法细节。

## 1) RelPose 主结果（室内）

### ScanNet（ATE，越低越好）
- `cut3r`: **0.817**
- `ttt3r`: **0.406**
- `ttt3r_random (p=0.5)`: **0.280**
- `ttt3r_momentum_inv_t1`: **0.261**（当前最佳）

相对改进（`ttt3r_momentum_inv_t1`）：
- 对 `cut3r`: **-68.0%**
- 对 `ttt3r`: **-35.6%**
- 对 `ttt3r_random`: **-6.8%**

### TUM（ATE，越低越好）
- `cut3r`: **0.166**
- `ttt3r`: **0.103**
- `ttt3r_random (p=0.5)`: **0.079**
- `ttt3r_momentum_inv_t1`: **0.063**（当前最佳）

相对改进（`ttt3r_momentum_inv_t1`）：
- 对 `cut3r`: **-61.8%**
- 对 `ttt3r`: **-38.2%**
- 对 `ttt3r_random`: **-20.3%**

## 2) KITTI 户外深度结果（bugfix 后）

数据位置：
- `eval_results_export/video_depth/kitti_s1_500_bugfix_final/`

### metric 对齐
- `ttt3r`: Abs Rel **0.128815**, Log RMSE **0.180974**, δ<1.25 **0.850601**
- `ttt3r_momentum_inv_t1`: Abs Rel **0.115049**, Log RMSE **0.171253**, δ<1.25 **0.866680**
- Abs Rel 相对改进：**-10.69%**

### scale 对齐
- `ttt3r`: Abs Rel **0.125868**, Log RMSE **0.173581**, δ<1.25 **0.867252**
- `ttt3r_momentum_inv_t1`: Abs Rel **0.118438**, Log RMSE **0.165685**, δ<1.25 **0.880861**
- Abs Rel 相对改进：**-5.90%**

### scale&shift 对齐
- `ttt3r`: Abs Rel **0.116942**, Log RMSE **0.171391**, δ<1.25 **0.873662**
- `ttt3r_momentum_inv_t1`: Abs Rel **0.106303**, Log RMSE **0.162461**, δ<1.25 **0.889503**
- Abs Rel 相对改进：**-9.10%**

## 3) 结论（仅结果层面）

- 当前最稳定、最优配置是：`ttt3r_momentum_inv_t1`。
- 室内（ScanNet/TUM）和户外（KITTI）上都取得了稳定提升。
- 目前结果已支持“brake 模块有效”的主结论。
