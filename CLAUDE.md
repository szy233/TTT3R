# TTT3R - zjc Branch Experiment Log

## Project Focus

This branch is centered on one main story:

- recurrent 3D reconstruction suffers from **systematic over-update**
- constant dampening already helps, which suggests the failure is not rare noise but a persistent update bias
- the strongest current method is the **Stability Brake**

Main formulation:

- `alpha_t = sigmoid(-tau * cos(delta_t, delta_{t-1}))`

Interpretation:

- high cosine alignment means consecutive updates are pointing in the same direction, so the model is likely re-updating an already converging state
- low cosine alignment means the state trajectory changes direction, so new information is likely entering and the brake should release

Current branch decision:

- keep the story focused on **brake only**
- do **not** make `geo gate` the main method

## Current Main Claim

The current evidence supports the following paper story:

1. recurrent state updates in CUT3R/TTT3R can over-react over long videos
2. train-free adaptive dampening in state space is enough to improve stability
3. the best current train-free variant is `ttt3r_momentum_inv_t1`
4. `geo gate` is not robust enough to be a main contribution

## Core Results Already Available

### 1. Formal RelPose Export

Exported from `origin/szy` into `eval_results_export/relpose/`.

Datasets and number of exported configs:

- `ScanNet`: 22 configs
- `TUM`: 12 configs
- `Sintel`: 3 configs

Best current brake-related result:

- `scannet_s3_1000 / ttt3r_momentum_inv_t1`: mean ATE `0.26147`, median ATE `0.19217`
- `tum_s1_1000 / ttt3r_momentum_inv_t1`: mean ATE `0.06339`, median ATE `0.06554`

Against constant dampening baseline `ttt3r_random`:

- `ScanNet`: mean ATE `0.27965 -> 0.26147` (`-6.50%`), median ATE `0.20304 -> 0.19217` (`-5.35%`)
- `TUM`: mean ATE `0.07900 -> 0.06339` (`-19.76%`), median ATE `0.08224 -> 0.06554` (`-20.30%`)

Against plain `cut3r`:

- `ScanNet`: mean ATE `0.81687 -> 0.26147` (`-67.99%`)
- `TUM`: mean ATE `0.16556 -> 0.06339` (`-61.71%`)

Important qualitative takeaway:

- `ttt3r_momentum_inv_t1` is clearly the strongest brake-style candidate
- `ttt3r_brake_geo` is not consistently good enough to keep as the main story

### 2. A3 Per-Scene Analysis

Generated locally from exported relpose logs.

Key comparisons:

- `ScanNet, ttt3r_random vs ttt3r_momentum_inv_t1`
  - common scenes: `65`
  - improved scenes: `31`
  - degraded scenes: `34`
  - median ATE: `0.20304 -> 0.19217`
  - mean relative improvement: `+0.92%`

- `TUM, ttt3r_random vs ttt3r_momentum_inv_t1`
  - common scenes: `8`
  - improved scenes: `7`
  - degraded scenes: `1`
  - median ATE: `0.08224 -> 0.065545`
  - mean relative improvement: `+14.90%`

- `ScanNet, ttt3r_random vs ttt3r_brake_geo`
  - common scenes: `65`
  - improved scenes: `20`
  - degraded scenes: `45`
  - mean relative improvement: `-35.37%`

- `TUM, ttt3r_random vs ttt3r_brake_geo`
  - common scenes: `8`
  - improved scenes: `5`
  - degraded scenes: `3`
  - mean relative improvement: `+3.84%`

Conclusion from A3:

- the brake-only direction is much stronger than `brake_geo`
- TUM shows the cleanest improvement pattern
- ScanNet is more mixed scene-by-scene, but still favors `momentum_inv_t1` in aggregate

### 3. S3 Tau Sensitivity

Currently available exported tau points:

- `tau = 1`
- `tau = 2`

Results:

- `ScanNet`
  - `tau=1`: mean ATE `0.26147`, median ATE `0.19217`
  - `tau=2`: mean ATE `0.31068`, median ATE `0.26213`

- `TUM`
  - `tau=1`: mean ATE `0.06339`, median ATE `0.06554`
  - `tau=2`: mean ATE `0.08219`, median ATE `0.05592`

Interpretation:

- `tau=1` is the safest default overall
- `tau=2` is too aggressive on ScanNet
- TUM median looks better at `tau=2`, but mean becomes worse
- current paper choice should remain `tau=1`

### 4. Video Depth Export

Exported datasets:

- `KITTI`
- `Bonn`
- `Sintel`

Original exported benchmark:

- `KITTI / metric`
  - `cut3r`: Abs Rel `0.15153`
  - `ttt3r`: Abs Rel `0.13192`
  - `ttt3r_joint`: Abs Rel `0.13437`

- `Bonn / metric`
  - `cut3r`: Abs Rel `0.09900`
  - `ttt3r`: Abs Rel `0.09974`
  - `ttt3r_joint`: Abs Rel `0.09408`

- `Sintel / metric`
  - `cut3r`: Abs Rel `1.02167`
  - `ttt3r`: Abs Rel `0.97764`
  - `ttt3r_joint`: Abs Rel `0.91725`

Takeaway from the original export:

- `TTT3R` already improves over `CUT3R`
- `ttt3r_joint` is competitive in video depth, but this is not yet the final brake-only story

### 5. KITTI Outdoor Brake Validation

This is the most important new result completed on `zjc`.

#### Initial problem

The first `kitti_s1_500` comparison between `ttt3r` and `ttt3r_momentum_inv_t1` produced identical outputs, which was treated as a bug signal.

#### Root cause

In `src/dust3r/model.py`, brake-side state was being reset whenever `reset_mask` existed, instead of only when a real reset happened.

Effect:

- brake state was cleared every step
- `_stability_brake()` repeatedly saw no valid previous delta
- `ttt3r_momentum_inv_t1` collapsed to plain `ttt3r`

#### Fix

Implemented:

- `has_reset = reset_mask is not None and bool(torch.any(reset_mask).item())`

Applied to:

- `_forward_impl`
- `forward_recurrent_lighter`
- `forward_recurrent_analysis`

Fix commit:

- `4e3e14e` - `fix: only reset brake state on true reset mask`

#### Final post-bugfix outdoor result

Stored under:

- `eval_results_export/video_depth/kitti_s1_500_bugfix_final/`

`metric` alignment:

- `ttt3r`: Abs Rel `0.128815`, Log RMSE `0.180974`, `delta<1.25 = 0.850601`
- `ttt3r_momentum_inv_t1`: Abs Rel `0.115049`, Log RMSE `0.171253`, `delta<1.25 = 0.866680`

`scale` alignment:

- `ttt3r`: Abs Rel `0.125868`, Log RMSE `0.173581`, `delta<1.25 = 0.867252`
- `ttt3r_momentum_inv_t1`: Abs Rel `0.118438`, Log RMSE `0.165685`, `delta<1.25 = 0.880861`

`scale&shift` alignment:

- `ttt3r`: Abs Rel `0.116942`, Log RMSE `0.171391`, `delta<1.25 = 0.873662`
- `ttt3r_momentum_inv_t1`: Abs Rel `0.106303`, Log RMSE `0.162461`, `delta<1.25 = 0.889503`

Relative improvement in Abs Rel:

- `metric`: `-10.69%`
- `scale`: `-5.90%`
- `scale&shift`: `-9.10%`

Key conclusion:

- after the reset bug is fixed, the brake path is clearly active
- brake improves outdoor depth metrics on KITTI
- this is the cleanest outdoor evidence currently available for the paper story

### 6. 7Scenes Video Reconstruction

Parsed from `eval_results_export/video_recon/7scenes_200/*/7scenes/logs_all.txt`.

Mean values:

- `cut3r`: acc `0.092`, comp `0.048`, nc1 `0.582`, nc2 `0.545`
- `ttt3r`: acc `0.027`, comp `0.023`, nc1 `0.600`, nc2 `0.561`
- `ttt3r_joint`: acc `0.021`, comp `0.022`, nc1 `0.594`, nc2 `0.565`

Takeaway:

- recurrent tuning brings a very large reconstruction accuracy gain over `cut3r`
- `ttt3r_joint` is strongest on acc/comp, but brake-only remains the cleaner paper narrative

### 7. A4 State Convergence Evidence

Local CO3D sequences used:

- `apple`
- `bottle`

Findings:

- `apple`
  - `cut3r`: mean delta norm `157.63`, last delta norm `155.86`, mean cosine `0.0518`
  - `ttt3r`: mean delta norm `95.89`, last delta norm `84.15`, mean cosine `0.4283`
  - mean update magnitude reduced by `39.2%`
  - final-step update magnitude reduced by `46.0%`

- `bottle`
  - `cut3r`: mean delta norm `251.77`, last delta norm `247.54`, mean cosine `0.2186`
  - `ttt3r`: mean delta norm `126.87`, last delta norm `117.92`, mean cosine `0.4917`
  - mean update magnitude reduced by `49.6%`
  - final-step update magnitude reduced by `52.4%`

Interpretation:

- the gated update path produces smaller and more directionally consistent state changes
- this supports the convergence-based explanation behind the stability brake

### 8. Single-Object Local Benchmark

Implemented and organized under `benchmark_single_object/`.

Available outputs include:

- sequence preparation scripts
- run scripts
- metric summaries
- runtime plots
- local CO3D benchmark outputs

Current lightweight benchmark observations:

- at length `12`, `ttt3r` is slightly faster than `cut3r`
- at length `24`, runtime is roughly comparable
- this benchmark is useful as a controlled local sanity check, but it is not yet a formal paper benchmark

The richer local export under `outputs_wsl_cpu/metrics/per_sequence_results_key_metrics.csv` is now preserved for later use.

## Best Paper Story Right Now

The strongest current narrative is:

1. recurrent 3D reconstruction suffers from over-update
2. constant dampening helps because it partially suppresses this error accumulation
3. adaptive dampening based on state-trajectory alignment is a natural train-free extension
4. `ttt3r_momentum_inv_t1` is the strongest current implementation
5. the outdoor KITTI bugfix result is the key new validation that this is not only an indoor relpose effect

## What Is Finished On zjc

1. Synced formal evaluation exports from `szy`
2. Built local A3 per-scene comparison summaries
3. Built local S3 tau sensitivity summaries
4. Wrote export summaries into `analysis_results/`
5. Fixed the brake reset bug in recurrent inference
6. Re-ran and validated post-fix KITTI outdoor depth results
7. Pulled full `kitti_s1_500_bugfix_final` outputs from the server into local `TTT3R`
8. Organized local A4 convergence evidence
9. Preserved single-object benchmark code and outputs

## What Should Be Done Next

### P0

1. Turn the current results into paper-quality plots and combined tables
2. Keep the method section centered on `stability brake` and drop `geo gate` from the main contribution
3. Add one polished summary figure showing relpose + KITTI + A4 consistency evidence together

### P1

1. Run missing tau points `0.5`, `1.5`, `3.0`
2. Add overhead analysis for brake computation itself
3. Upgrade A4 from local proxy evidence to a more formal benchmark setting

### P2

1. Consolidate all final numbers into one camera-ready result sheet
2. Write the abstract/introduction around the over-update story
3. Decide whether any full output folders should be uploaded to GitHub or only summarized artifacts

## Important Cautions

- `analysis_results/` is gitignored unless explicitly force-added
- the worktree still contains many large untracked result folders; avoid accidental bulk commits
- the full `kitti_s1_500_bugfix_final` directory is large, so upload it only if really needed
- do not over-claim tau sensitivity before more points are run

## Branch Record

- Branch: `zjc`
- Main brake bugfix commit: `4e3e14e`
- KITTI bugfix final metrics commit: `4f1a9b2`
- Source of formal exported logs: `origin/szy`
