# E8 — direction on ARKitScenes: rotation drift and sequence length

> **Fixed before any E8 value existed.** Its SHA-256 is written to `results/e8_registration.sha256`, and `scripts/run_e8.sh` refuses to start if the file no longer matches. Nothing above the first line holding only `---` may be edited after results exist. Corrections and results go in dated sections after it.

Submission: ICLR 2027, 28147. Registered 2026-11-08. Follows `PREREG_E7.md`, whose comparison rule, arm definitions and code apply unchanged.

## Why this exists

E6 and E7 found one robust effect of the decomposed direction: on KITTI Odometry it lowers r_rel, a rotation error accumulated over 100 to 800 m, under both ortho's own norm schedule and const α's norm, while frame-to-frame RPE rotation does not improve. On ARKitScenes 500f the same direction raises ATE or ties it. Two readings of that contrast are open:

| Reading | What would show it |
|---|---|
| Direction lowers accumulated rotation drift on ARKitScenes too, and ATE there is driven by something else | A rotation error over long frame gaps falls with direction on ARKitScenes |
| ARKitScenes 500f is too short for the effect, which needs long sequences | Direction's ATE cost on ARKitScenes shrinks from 500 to 1000 frames |

Neither has been measured. The paper and replies currently say only that direction ties or hurts ARKitScenes ATE.

## What was already known at registration

This registration is not blind to the reference arms. Read before writing it:

- ARKitScenes 500f, 18 scenes, means of ATE RMSE and of RPE translation and rotation at a 1-frame gap, for `ortho_check`, `normreplay`, `const010/020/033`, `dir010/020/033`, TTT3R, TTSA3R, brake τ=1 and const 0.5 (`results/e6.txt`, `results/e7.txt`). Per-scene values for the 8 scenes used in E8b were not read.
- The same metrics on TUM 1000f, ScanNet 1000f and KITTI, and KITTI r_rel and t_rel, for the E6 and E7 arms.
- The rounding-noise arms `orthop` and `const033p` on TUM and KITTI.

**Nothing exists for:** any rotation error at a gap above 1 frame on any dataset, and any run on ARKitScenes at 1000 frames.

## Protocol

**Code.** Unchanged from E7. `scripts/run_e8.sh` checks that `/usr/bin/git diff 43f7e29 -- src eval` still hashes to the value in `results/e7_code.sha256`, and refuses to start otherwise.

**New metric, R_Δ.** For each sequence, the RPE rotation angle in degrees at a gap of Δ frames, computed by the same call as the existing RPE rotation in `eval/relpose/evo_utils.py::eval_metrics` (evo `main_rpe.rpe`, `PoseRelation.rotation_angle_deg`, `align=True`, `correct_scale=True`, `delta_unit=Unit.frames`, `rel_delta_tol=0.01`, `all_pairs=True`, RMSE), with `delta=Δ` in place of 1. Predictions come from the saved `pred_traj.txt` and the dataset's ground-truth file, read as `eval/relpose/launch.py` reads them, with predicted timestamps replaced by ground-truth timestamps when the counts match. The script is `scripts/e8_rot_drift.py`. **Primary: R_100.** R_50 and R_200 are reported beside it with no prediction.

**Comparison rule.** From `scripts/stats_rule.py`, as in E7. "A is better than B" needs a lower mean, p < 0.05 and a gap above 2.2%. Anything else is "tied", with the CI beside it. Holm-adjusted p within each dataset is reported beside the binding rule. A gap smaller than the rounding noise measured below is described as within rounding noise, whatever the rule says.

**Designs.** "Design A" is `ortho_check` against `normreplay`, direction at ortho's own norm schedule. "Design B at α" is `dirNNN` against `constNNN`, direction at const α's norm.

### E8a — R_Δ on existing trajectories, no GPU

| Dataset | Design A | Design B scales | Role |
|---|---|---|---|
| ARKitScenes 500f, 18 scenes | yes | 0.10, 0.20, 0.33 | primary |
| TUM 1000f, 8 sequences | yes | 0.10, 0.20, 0.33 | secondary |
| ScanNet 1000f, the 65 valid scenes | yes | 0.10, 0.33 | secondary |
| KITTI Odometry, 11 sequences | yes | 0.10, 0.20, 0.33 | instrument check I2 |

Rounding noise on R_100: `orthop` against `ortho_check` on TUM and KITTI.

### E8b — ARKitScenes at 1000 frames, new runs

The 8 scenes with 1000 prepared frames: 41069021, 41069025, 41069046, 41125696, 41125700, 41125709, 41125718, 41125722. Dataset key `arkit_s1_1000`, output group `arkit_s1_1000__eight`.

Arms, all defined as in `scripts/run_arm.sh`: `ortho_check` (logs norms), `normreplay` (replays them), `dir033`, `const033`, `dir010`, `const010`, `orthop` (rounding noise on ARKitScenes), then `ttt3r_plain` and `brake_t1` for context with no prediction.

Metrics: ATE RMSE (primary), R_100, and the 1-frame RPEs. The same 8 scenes at 500 frames are taken from the existing E6 and E7 arms.

### Checks

- **I1, loader.** Before any R_Δ above 1 is printed, R_1 recomputed by `e8_rot_drift.py` must match the logged per-sequence "RPE rot" within 0.5% for every sequence of ARKitScenes `dir010`, TUM `ortho_check`, ScanNet `ortho_check` and KITTI `ortho_check`. A miss stops E8a until the loader is fixed. The check output goes to `results/e8_instrument.txt`.
- **I2, does R_100 track the KITTI effect.** On KITTI, r_rel says direction is lower in all four cells: design A and design B at 0.10, 0.20 and 0.33. I2 passes if R_100's mean difference has the same sign in all four. If it fails, R_100 is not treated as an analogue of r_rel: E8a is still reported in full, but no text may say that ARKitScenes, TUM or ScanNet show "the KITTI effect".
- **I3, gain.** The logged gain of each `dirNNN` arm in E8b equals its scale within 0.5%, as in E7 check 3.
- **Sequence counts.** A sequence whose prediction and ground truth differ in length is reported and left out of that dataset's tests. Nothing else is excluded.

## Predictions

These follow the KITTI pattern: direction lowers accumulated rotation drift, and its benefit grows with length.

- **P8a.** On ARKitScenes 500f, R_100 is better under design A.
- **P8b.** On ARKitScenes 500f, R_100 is better under design B at no fewer than 2 of the 3 scales.
- **P8c.** On ARKitScenes 1000f, R_100 is better under design A and under design B at both 0.10 and 0.33.
- **P8d.** On the 8 scenes, direction's relative ATE cost shrinks from 500 to 1000 frames. For each cell (A, B at 0.10, B at 0.33) and scene s, d_s = ln(ATE_direction / ATE_reference) at 1000 minus the same at 500. A cell supports P8d when the mean of d_s is below −0.022 and the Wilcoxon p over the 8 scenes is below 0.05. P8d holds when no fewer than 2 of the 3 cells support it.
- **TUM and ScanNet.** No prediction. Reported in full.

ATE at 1000 frames, each design cell and each arm against `ortho_check`, is reported with no prediction.

## Consequences for the replies and the revision

| Outcome | Consequence |
|---|---|
| I2 passes, P8a and P8b hold | May say that on ARKitScenes, as on KITTI, the decomposed direction lowers rotation error over 100 frames, while ATE does not improve |
| I2 passes, P8a or P8b fails | Must not say ARKitScenes shares the KITTI rotation effect. Report each cell |
| I2 fails | Report R_100 as a rotation error over 100 frames only, never as the KITTI effect |
| P8c holds | May add "and at 1000 frames on 8 scenes" to whichever statement above is allowed |
| P8d holds | May say that on ARKitScenes the ATE cost of direction shrinks from 500 to 1000 frames, on 8 scenes. Must not say direction helps long sequences in general |
| P8d fails | No statement that the effect depends on length |
| Direction better on ARKitScenes 1000f ATE in no fewer than 2 of the 3 cells | Not predicted. Report it as not predicted, with its n=8 |
| R_100 better under design A on all four datasets, and under design B at no fewer than 2 of 3 scales on ARKitScenes and TUM and at both scales on ScanNet | May say that across the four datasets direction lowers rotation error over 100 frames |
| Any other pattern on TUM or ScanNet | Report each cell, and draw no general statement |

Under every outcome: the replies must not say that direction improves ARKitScenes accuracy unless ATE is better by the common rule. Const 0.10 remains not worse than ortho in ATE on any dataset, as E7 found, and any text on E8 sits beside that sentence.

**Default.** Nothing here changes the default configuration or makes any arm the paper's method.

## Budget and stopping

E8a takes minutes of CPU. E8b is 9 arms of about 8000 frames each. On the E7 timings, about 0.5 to 1 hour per arm on a shared GPU, so about 5 to 9 GPU-hours, run as two workers on GPUs 0 and 1, about 3 to 5 hours of wall clock. If E8b has not finished by 2026-11-10 12:00 CST, stop, rule on the arms that finished, and report the rest as not run.

---

## Deviation, 2026-11-08, during the first E8a run

Everything above the line is the registered text, unchanged. The first E8a run stopped on ScanNet at R_200, after the KITTI, ARKitScenes and TUM blocks had printed, because two scenes, scene0718_00 and scene0801_00, have too few frames for a 200-frame gap. The registration covered length mismatches but not this. The script now leaves a sequence out of only the gap it is too short for, and prints each case. ScanNet R_100 and R_50 use all 65 scenes. ScanNet R_200, which carries no prediction, uses 63. No other dataset is affected.

## Results, 2026-11-08, E8a

Numbers come from `results/e8a.txt`. Check I1 passed: R_1 recomputed from the saved trajectories matches the logged RPE rotation within 0.0017% on every sequence (`results/e8_instrument.txt`).

**R_100, direction against reference, mean (gap, direction wins, p, verdict).**

| Dataset | Design A | B at 0.10 | B at 0.20 | B at 0.33 |
|---|---|---|---|---|
| KITTI, n=11 | 28.98 vs 36.99 (−21.6%, 10/11, p=0.024, better) | 25.43 vs 34.58 (−26.5%, 8/11, p=0.014, better) | 24.74 vs 38.63 (−36.0%, 8/11, p=0.032, better) | 25.62 vs 37.93 (−32.4%, 11/11, p=0.001, better) |
| ARKitScenes 500f, n=18 | 45.82 vs 47.86 (−4.3%, 10/18, p=0.62, tied) | 67.76 vs 61.13 (+10.8%, 2/18, p=0.003, worse) | 61.00 vs 53.13 (+14.8%, 4/18, p=0.012, worse) | 54.03 vs 47.58 (+13.6%, 4/18, p=0.022, worse) |
| TUM 1000f, n=8 | 5.19 vs 2.62 (+97.7%, 5/8, p=0.84, tied) | 16.60 vs 12.06 (+37.6%, 1/8, p=0.039, worse) | 12.06 vs 9.25 (+30.4%, 3/8, p=0.15, tied) | 8.00 vs 4.45 (+79.7%, 3/8, p=0.25, tied) |
| ScanNet 1000f, n=65 | 13.80 vs 8.35 (+65.3%, 29/65, p=0.072, tied) | 35.97 vs 17.05 (+111.0%, 8/65, p<0.001, worse) | not run | 16.97 vs 7.62 (+122.7%, 17/65, p<0.001, worse) |

Rounding noise on R_100: `orthop` against `ortho_check` moves the mean by +4.4% on TUM and −7.1% on KITTI. Every "better" and "worse" above is larger than both.

**I2 passes.** R_100 falls with direction in all four KITTI cells, as r_rel does, so it tracks the KITTI effect.

**P8a fails** (tied) and **P8b fails**: at const α's norm, direction raises R_100 on ARKitScenes at all three scales. The general four-dataset statement is not allowed.

**Ruling.** This is the row "I2 passes, P8a or P8b fails". The replies and the revision must not say that ARKitScenes shares the KITTI rotation effect, and must report each cell. Read with TUM and ScanNet, the pattern is that the rotation-drift reduction appears on KITTI only, and at a flat norm direction raises rotation error over 100 frames on all three indoor datasets where the rule can tell (ARKitScenes at every scale, ScanNet at both, TUM at 0.10). On TUM, ortho's lower ATE against same norm comes with a higher mean R_100, tied by the rule.

## Results, 2026-11-08, E8b

Numbers come from `results/e8b.txt`. All nine arms finished by 13:39 with exit 0 on all 8 scenes, well inside the stop date, and no sequence was left out for a length mismatch. Check I3 passed: the logged gain of `dir033` and `dir010` equals its scale in all 8 logs (largest gap 0.0000%). The first `e8b` output did not print the 1-frame RPEs listed under Metrics. The script now prints them, and the rest of its output is unchanged.

Rounding noise on ARKitScenes 1000f: `orthop` against `ortho_check` moves mean ATE by +0.2% and mean R_100 by −1.0%. This is the only rounding-noise measurement on ARKitScenes, on these 8 scenes at 1000 frames. ARKitScenes 500f still has none.

**ARKitScenes 1000f, n=8, direction against reference, mean (gap, direction wins, p, verdict).**

| Metric | Design A | B at 0.10 | B at 0.33 |
|---|---|---|---|
| ATE RMSE | 1.1370 vs 0.9809 (+15.9%, 5/8, p=0.95, tied) | 1.1807 vs 1.0395 (+13.6%, 2/8, p=0.055, tied) | 1.1452 vs 0.9296 (+23.2%, 3/8, p=0.25, tied) |
| R_100 | 60.17 vs 56.17 (+7.1%, 2/8, p=0.078, tied) | 72.18 vs 67.47 (+7.0%, 3/8, p=0.15, tied) | 61.55 vs 59.21 (+4.0%, 4/8, p=0.55, tied) |
| R_50 | 42.74 vs 38.75 (+10.3%, 3/8, p=0.25, tied) | 60.67 vs 53.45 (+13.5%, 2/8, p=0.039, worse) | 44.47 vs 43.58 (+2.0%, 5/8, p=0.95, tied) |
| R_200 | 78.77 vs 77.73 (+1.3%, 2/8, p=0.25, tied) | 87.72 vs 85.68 (+2.4%, 3/8, p=0.46, tied) | 80.83 vs 79.42 (+1.8%, 6/8, p=0.74, tied) |
| RPE trans, 1 frame | 0.08883 vs 0.06396 (+38.9%, 1/8, p=0.016, worse) | 0.16077 vs 0.15424 (+4.2%, 4/8, p=0.84, tied) | 0.10929 vs 0.07655 (+42.8%, 0/8, p=0.0078, worse) |
| RPE rot, 1 frame | 8.25 vs 5.20 (+58.6%, 2/8, p=0.055, tied) | 17.56 vs 13.87 (+26.6%, 0/8, p=0.0078, worse) | 9.72 vs 9.91 (−2.0%, 3/8, p=0.95, tied) |

Holm-adjusted p at R_100: 0.23 (A), 0.30 (B at 0.10), 0.55 (B at 0.33). At n=8 the Wilcoxon p cannot fall below 0.0078. The bootstrap CI excludes zero on the side where direction is worse for ATE at B 0.10, for R_100 at A and B 0.10, and for 1-frame RPE rotation at A, but p is above 0.05 in each, so the rule calls them tied.

**P8c fails**: all three cells are tied, and direction's mean R_100 is higher in all three, by more than the rounding noise.

**P8d fails** (0 of 3 cells). On the 8 scenes, mean ln(ATE_direction / ATE_reference) is +0.172 (A), +0.201 (B at 0.10) and +0.235 (B at 0.33) at 500 frames, and +0.204, +0.161 and +0.279 at 1000. The changes are +0.032, −0.040 and +0.044, none with p below 0.05.

Direction is better in ATE in 0 of 3 cells. Against `ortho_check`, every arm ties in ATE by the rule.

Post hoc, not registered (`results/e8_posthoc.txt`, from `e8_rot_drift.py posthoc`): against TTT3R at 1000 frames on the 8 scenes, brake (0.937 vs 1.030, 5/8, p=0.38), const 0.33, const 0.10, ortho and same norm all tie in ATE. Brake's 1-frame RPE rotation is higher than TTT3R's (9.89 vs 4.97, 0/8, p=0.0078).

**Ruling.** Rows "P8d fails" and, with E8a, "I2 passes, P8a or P8b fails". P8c fails, so nothing may be extended "to 1000 frames". No text may say that direction's effect depends on sequence length, or that direction improves ARKitScenes accuracy or lowers its rotation drift at 500 or 1000 frames. At n=8 these tests have little power, so "tied" at 1000 frames is not evidence that direction is harmless there. Every design-cell mean is worse with direction: ATE by 13.6% to 23.2%, R_100 by 4.0% to 7.1%. Taken with E8a, the reduction of rotation drift by direction is found on KITTI Odometry only. Const 0.10 still ties ortho in ATE on ARKitScenes 1000f (1.0395 vs 1.1370, p=1.00), so E7's sentence that const 0.10 is not worse than ortho holds there too.
