# Reproducibility Study (SAFE224, Stability Brake)

## Goal
Run the same configuration 3 times and report mean ± std to verify that improvements are not accidental.

## Setup
- Branch: `zjc-ttt3r-sensitivity-clean`
- Dataset split: sampled single-object sequences (`apple/bottle`, lengths `12/24`)
- Method: `ttt3r_momentum_inv_t1`
- Image size: `224` (GPU-safe on 8GB VRAM)
- Device: `cuda`
- Repeats:
  - Fixed-seed group: `42, 42, 42`
  - Different-seed group: `41, 42, 43`

## Command
```bash
python3 benchmark_single_object/scripts/run_reproducibility.py \
  --config benchmark_single_object/configs/experiment_brake_ablation_safe224.yaml \
  --methods ttt3r_momentum_inv_t1 \
  --tag repro_safe224_seedstudy
```

## Outputs
- Raw runs:
  - `benchmark_single_object/outputs_ablation_safe/metrics/repro_safe224_seedstudy/repro_raw_results.csv`
- Summary (mean ± std):
  - `benchmark_single_object/outputs_ablation_safe/metrics/repro_safe224_seedstudy/repro_summary_overall.csv`
  - `benchmark_single_object/outputs_ablation_safe/metrics/repro_safe224_seedstudy/repro_summary_by_length.csv`
- Per-sequence stability:
  - `benchmark_single_object/outputs_ablation_safe/metrics/repro_safe224_seedstudy/repro_sequence_stability.csv`

## Key Result
- `basic_consistency_score`:  
  - fixed seed: `1.1085 ± 0.5101`
  - different seeds: `1.1085 ± 0.5101`
- `loop_closure_trans_error`:  
  - fixed seed: `0.3946 ± 0.3710`
  - different seeds: `0.3946 ± 0.3710`

Across repeated runs, the per-sequence std for geometric quality metrics is `0.0` in both groups, indicating deterministic/stable inference under SAFE224 settings.
