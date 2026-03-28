# S2 Overhead Comparison: `drift>0` vs `drift0` (SAFE224, Local)

## Experiment Scope
- Platform: local PC (RTX 4060 Laptop, 8GB)
- Setting: SAFE224, sampled single-object sequences (`len=12/24`)
- Repeats: fixed-seed (42,42,42) + different-seed (41,42,43)
- Methods:
  - `ttt3r_momentum_inv_t1` (`alpha_drift=0.15`)
  - `ttt3r_momentum_inv_t1_drift0` (`alpha_drift=0.0`)

## Raw Output Paths
- `benchmark_single_object/outputs_ablation_safe/metrics/repro_safe224_drift_compare/repro_raw_results.csv`
- `benchmark_single_object/outputs_ablation_safe/metrics/repro_safe224_drift_compare/repro_summary_overall.csv`
- `benchmark_single_object/outputs_ablation_safe/metrics/repro_safe224_drift_compare/repro_summary_by_length.csv`

## Overhead Results (mean ± std)

### Overall
| Protocol | Method | Runtime (s) | Per-frame (s) |
|---|---|---:|---:|
| fixed_seed | `ttt3r_momentum_inv_t1` | `15.6084 ± 1.7761` | `0.9508 ± 0.2819` |
| fixed_seed | `ttt3r_momentum_inv_t1_drift0` | `15.3243 ± 1.3091` | `0.9327 ± 0.2562` |
| different_seed | `ttt3r_momentum_inv_t1` | `14.9859 ± 1.4224` | `0.9086 ± 0.2386` |
| different_seed | `ttt3r_momentum_inv_t1_drift0` | `14.9558 ± 1.3446` | `0.9084 ± 0.2436` |

### By Length (different_seed)
| Method | len=12 per-frame (s) | len=24 per-frame (s) |
|---|---:|---:|
| `ttt3r_momentum_inv_t1` | `1.1367 ± 0.0152` | `0.6805 ± 0.0110` |
| `ttt3r_momentum_inv_t1_drift0` | `1.1410 ± 0.0247` | `0.6758 ± 0.0096` |

## Quality Snapshot (overall mean)
| Method | Basic Consistency | Loop Trans Error |
|---|---:|---:|
| `ttt3r_momentum_inv_t1` | `1.1085` | `0.3946` |
| `ttt3r_momentum_inv_t1_drift0` | `1.1083` | `0.3903` |

## Interpretation
1. **Runtime overhead is essentially the same** between `drift>0` and `drift0` in this local SAFE224 setup.
2. **No meaningful latency penalty** from keeping non-zero drift (`alpha_drift=0.15`).
3. On this small local subset, quality difference is very small; this table is mainly an **overhead report**, not final global-quality evidence.

## Conclusion for Current Local Stage
For local reproducibility and efficiency claims, we can safely state:  
**`alpha_drift` setting does not materially change inference overhead under SAFE224.**
