# TTT3R — zjc Branch Working Notes

## Project Goal

NeurIPS-style project log for train-free, inference-time state dampening in recurrent 3D reconstruction.  
Current branch focus: organize exported evaluation results, formalize the stability-brake story, and keep a clean local record for follow-up experiments on `zjc`.

## Current Position

The main story on this branch is no longer "frequency gating".  
The strongest validated direction is:

- **Stability Brake**: `alpha_t = sigmoid(-tau * cos(delta_t, delta_{t-1}))`
- Problem framing: **systematic over-update** in recurrent state updates
- Core claim: adaptive dampening is better than constant dampening when scene dynamics vary over time

The exported formal results were synced from `origin/szy` into `eval_results_export/`, then summarized and visualized locally on this branch.

## Main Files on zjc

| File | Purpose |
|------|---------|
| `analysis/per_scene_improvement_analysis.py` | A3 per-scene relpose comparison |
| `analysis/s3_brake_sensitivity.py` | S3 tau sensitivity summary |
| `analysis/state_convergence_analysis.py` | A4 state convergence logging and plots |
| `analysis/a3_per_scene_distribution.py` | Original per-scene plotting script from `szy` |
| `analysis_results/formal_export_summary.md` | Human-readable summary of exported results |
| `eval_results_export/` | Exported formal logs and metrics from `szy` branch |

## Exported Formal Results

### A3 Per-Scene Relpose

#### ScanNet: `ttt3r_random` vs `ttt3r_momentum_inv_t1`

- Common scenes: 65
- Improved scenes: 31
- Degraded scenes: 34
- Median ATE: `0.20304 -> 0.19217`
- Mean relative improvement: `+0.92%`
- Median relative improvement: `-1.35%`

#### TUM: `ttt3r_random` vs `ttt3r_momentum_inv_t1`

- Common scenes: 8
- Improved scenes: 7
- Degraded scenes: 1
- Median ATE: `0.08224 -> 0.065545`
- Mean relative improvement: `+14.90%`
- Median relative improvement: `+10.21%`

#### ScanNet: `ttt3r_random` vs `ttt3r_brake_geo`

- Common scenes: 65
- Improved scenes: 20
- Degraded scenes: 45
- Median ATE: `0.20304 -> 0.24746`
- Mean relative improvement: `-35.37%`
- Median relative improvement: `-21.93%`

#### TUM: `ttt3r_random` vs `ttt3r_brake_geo`

- Common scenes: 8
- Improved scenes: 5
- Degraded scenes: 3
- Median ATE: `0.08224 -> 0.054865`
- Mean relative improvement: `+3.84%`
- Median relative improvement: `+10.02%`

### Interpretation

1. `momentum_inv_t1` is clearly stronger than constant dampening on **TUM**.
2. On **ScanNet**, the improvement is weaker and more mixed scene-by-scene.
3. `brake_geo` does not behave like a universal improvement.
4. The current evidence supports **stability brake alone** more strongly than `brake + geo`.

## S3 Tau Sensitivity

Only exported `tau=1` and `tau=2` are currently available.

### ScanNet

- `tau=1`: median ATE `0.19217`, mean ATE `0.26147`
- `tau=2`: median ATE `0.26213`, mean ATE `0.31068`

### TUM

- `tau=1`: median ATE `0.065545`, mean ATE `0.06339`
- `tau=2`: median ATE `0.05592`, mean ATE `0.08219`

### Interpretation

- ScanNet currently favors **tau = 1**
- TUM shows mixed behavior: lower median at `tau=2`, but worse mean
- The present conclusion is still: **tau = 1 is the safer default**
- A real sensitivity section still needs more points: `0.5, 1.5, 3.0`

## Exported Video Depth

### KITTI

- `cut3r`: Abs Rel `0.15153`, RMSE `5.66694`, delta<1.25 `0.80434`
- `ttt3r`: Abs Rel `0.13192`, RMSE `5.42614`, delta<1.25 `0.86530`
- `ttt3r_joint`: Abs Rel `0.13437`, RMSE `5.38475`, delta<1.25 `0.85774`

### Bonn

- `cut3r`: Abs Rel `0.09900`, RMSE `0.34637`, delta<1.25 `0.90612`
- `ttt3r`: Abs Rel `0.09974`, RMSE `0.33887`, delta<1.25 `0.92143`
- `ttt3r_joint`: Abs Rel `0.09408`, RMSE `0.32358`, delta<1.25 `0.93431`

### Sintel

- `cut3r`: Abs Rel `1.02167`, RMSE `6.88020`, delta<1.25 `0.23766`
- `ttt3r`: Abs Rel `0.97764`, RMSE `6.67607`, delta<1.25 `0.23245`
- `ttt3r_joint`: Abs Rel `0.91725`, RMSE `6.54943`, delta<1.25 `0.24723`

## Exported 7scenes Reconstruction

Mean values parsed from `eval_results_export/video_recon/7scenes_200/*/7scenes/logs_all.txt`.

- `cut3r`: acc `0.092`, comp `0.048`, nc1 `0.582`, nc2 `0.545`
- `ttt3r`: acc `0.027`, comp `0.023`, nc1 `0.600`, nc2 `0.561`
- `ttt3r_joint`: acc `0.021`, comp `0.022`, nc1 `0.594`, nc2 `0.565`

## Narrative Draft

### Problem

Recurrent 3D reconstruction applies state updates too aggressively over long videos.  
Even when incoming frames carry limited new geometry, the recurrent state still updates with nearly the same strength.  
Constant dampening already helps a lot, which suggests that **over-update** is a central failure mode.

### Method

Use state-trajectory consistency as an online control signal:

`alpha_t = sigmoid(-tau * cos(delta_t, delta_{t-1}))`

- cosine high: updates are aligned, likely redundant, so brake harder
- cosine low: updates change direction, likely new information, so release the brake

### Why This Story Is Stronger

- It explains why constant `x0.5` works at all
- It naturally motivates adaptive dampening
- It aligns with the current theory direction: over-update bound, regret comparison, optimal tau
- It fits the empirical pattern: dynamic scenes benefit more than static scenes

## What Is Already Done on zjc

1. Imported exported formal logs from `szy` into `eval_results_export/`
2. Generated official local A3 figures for ScanNet/TUM
3. Generated local S3 tau summaries from available exported runs
4. Wrote a readable summary in `analysis_results/formal_export_summary.md`
5. Built and ran `A2` proxy analysis on local CO3D windows
6. Consolidated local A4 state-convergence results on CO3D apple/bottle sequences
7. Pushed these artifacts to branch `zjc`

## Suggested Next Steps

### P0

1. Finish the **brake-only** code path as the main method and stop extending `brake_geo`
2. Run **KITTI** outdoor validation for `ttt3r_momentum_inv_t1`
3. Turn current A3/S3 outputs into paper-quality combined figures

### P1

1. Upgrade **A2** from local proxy to formal relpose-based analysis
2. Upgrade **A4** from local CO3D evidence to formal relpose/video benchmark analysis
3. Run missing tau values: `0.5, 1.5, 3.0`

### P2

1. Write a polished abstract around over-update and adaptive dampening
2. Consolidate all result tables into one camera-ready summary sheet
3. Merge the useful parts of this note back into the final project `CLAUDE.md`

## Cautions

- `analysis_results/` is gitignored by default, so result directories need `git add -f` if they should be versioned
- The local worktree still contains unrelated modified files in `src/`; do not auto-commit them together with analysis artifacts
- Exported sensitivity is incomplete; avoid over-claiming the tau story until more points are run

## Branch Record

- Branch: `zjc`
- Export/artifact commit: `bfe6baa`
- Source of exported logs: `origin/szy`
