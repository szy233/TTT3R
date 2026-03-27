# KITTI Outdoor Validation (March 27, 2026)

This run compares `ttt3r` and `ttt3r_momentum_inv_t1` on `kitti_s1_500`.

Data/model:

- Dataset: `data_depth_selection.zip` (official KITTI depth selection), preprocessed to `data/long_kitti_s1`
- Eval split: `kitti_s1_500`
- Weight: `src/cut3r_512_dpt_4_64.pth`
- Image size: `512`

## Metrics

### `result_metric.json`

| model | Abs Rel | Sq Rel | RMSE | Log RMSE | δ < 1.25 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ttt3r` | 0.128815 | 0.912491 | 5.700562 | 0.180974 | 0.850601 |
| `ttt3r_momentum_inv_t1` | 0.128815 | 0.912491 | 5.700562 | 0.180974 | 0.850601 |

### `result_scale.json`

| model | Abs Rel | Sq Rel | RMSE | Log RMSE | δ < 1.25 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ttt3r` | 0.125868 | 0.853534 | 5.495092 | 0.173581 | 0.867252 |
| `ttt3r_momentum_inv_t1` | 0.125868 | 0.853534 | 5.495092 | 0.173581 | 0.867252 |

### `result_scale&shift.json`

| model | Abs Rel | Sq Rel | RMSE | Log RMSE | δ < 1.25 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ttt3r` | 0.116942 | 0.835753 | 5.547695 | 0.171391 | 0.873662 |
| `ttt3r_momentum_inv_t1` | 0.116942 | 0.835753 | 5.547695 | 0.171391 | 0.873662 |

## Conclusion

For this KITTI run, both models produced numerically identical metrics under all three align modes (`metric`, `scale`, `scale&shift`).

This strongly suggests the current video-depth path does not expose a measurable brake gain on this setting.
