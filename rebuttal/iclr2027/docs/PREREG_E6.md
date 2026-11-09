# E6 — predictions and decision rules

> **Fixed before any E6 arm produced an ATE.** Its SHA-256 is written to `results/e6_registration.sha256`, and `scripts/run_e6.sh` refuses to start if the file no longer matches. Nothing below may be edited after results exist. Corrections and results go in dated sections at the end.

Submission: ICLR 2027, 28147. Registered 2026-11-07. Follows `PREREG_E1_E5.md`, whose common protocol and comparison rule apply unchanged.

## Why this exists

The draft replies were about to say more for direction, using logged results that have no same-norm control. These runs test those statements before they are made:

| Statement under test | Current evidence | Experiment |
|---|---|---|
| The rotation gain on KITTI Odometry comes from direction | Ortho beats brake, const 0.5 and TTT3R on r_rel, with no same-norm arm | E6a |
| No flat step size reaches ortho on TUM | Only α=0.33 and α=0.5 have been run | E6b |
| The ScanNet loss belongs to the α∥=0.05 operating point | Logged α∥ sweep, read after the fact, with no same-norm arm | E6c |

## What was already known at registration

This registration is not blind to the reference arms. These values were read before writing it.

| Dataset | TTT3R | TTSA3R | Const 0.33 | Const 0.5 | Brake τ=1 | Ortho | Same norm |
|---|---|---|---|---|---|---|---|
| TUM 1000f, ATE | 0.10251 | 0.09075 | 0.06591, older code | 0.07859 | 0.06339 | 0.05454 | 0.07437 |
| ScanNet 1000f, ATE | 0.40576 | 0.37795 | 0.26237 | 0.28274 | 0.26147 | 0.48829 | 0.31858 |
| ARKitScenes 500f, ATE | 0.55804 | 0.55591 | not run | 0.52754 | 0.53907 | 0.60927 | 0.54458 |
| KITTI Odometry, r_rel (deg/100m) | 20.603 | no trajectories | not run | 17.817 | 17.661 | 11.881 | not run |
| KITTI Odometry, ATE RMSE | 180.23 | 174.30 | not run | 178.24 | 172.21 | 160.98 | not run |

Logged α∥ sweep, read from folders named `ddd3r_a10` to `ddd3r_a25`:

| α∥ | TUM | ScanNet |
|---|---|---|
| 0.05 | 0.05454 | 0.48829 |
| 0.10 | 0.05471 | 0.43715 |
| 0.15 | 0.05504 | 0.39886 |
| 0.20 | 0.05602 | 0.36667 |
| 0.25 | 0.06085 | 0.34364 |

**Nothing exists for:** the same-norm replay on KITTI, flat multipliers below 0.33 on TUM, the same-norm replay at any α∥ other than 0.05, or α∥=0.15 on ARKitScenes or KITTI. On KITTI, only γ variants have been run away from the default, and only on sequences 03 and 04.

**Why α∥=0.15.** It was chosen after reading the sweep above. It is the largest value that still wins 8/8 over brake on TUM and is no longer worse than TTT3R or TTSA3R on ScanNet. ARKitScenes and KITTI played no part in that choice, so they are the only out-of-sample tests here. This setting is tested as a sensitivity analysis. **No outcome below makes it the paper's default.**

## Common protocol

- Checkpoint `model/cut3r_512_dpt_4_64.pth`, `--size 512`, `--pose_only`, compiled RoPE. `git diff 43f7e29 -- src eval` is empty at registration, so the model code matches E1–E5.
- Comparison rule from `scripts/stats_rule.py`. "A is better than B" needs a lower mean, p < 0.05 and a gap above 2.2%. Anything else is "tied", and the CI is reported beside it.
- ScanNet runs on the same 65 valid scenes as E1b. No sequence is excluded after the fact.
- Every outcome is reported, whichever way it goes.

## E6a — same norm on KITTI Odometry

**Arms.** `ortho_check` logs the applied norm, and `normreplay` replays it along the raw delta. Both run on `kitti_odo_full`, 11 sequences.

**Check.** `ortho_check` must match the existing `ortho_plain` within 0.5% on every sequence. A miss blocks the ruling until explained.

**Primary metric.** r_rel from `eval/relpose/eval_kitti_odom.py`, because r_rel is the claim under test. That script aligns with Sim(3) and averages per-sequence means, so it is not the KITTI devkit metric and must not be called official. ATE RMSE, t_rel and both RPEs are reported beside it, with no prediction.

**Prediction P6a.** `ortho_check` is better than `normreplay` on r_rel.

| Outcome on r_rel | Meaning | Consequence for the replies |
|---|---|---|
| Ortho better | The rotation gain survives at identical norm | May say direction lowers segment rotation error on KITTI at identical norm |
| Tied | The gain cannot be credited to direction | Report r_rel against the scalars only, and state that the same-norm replay ties |
| Same norm better | The gain comes from the norm schedule, not the direction | Do not offer r_rel as evidence for direction |

## E6b — flat multipliers on TUM

**Arms.** `const010`, `const015`, `const020`, `const025`, `const033`, which are `ddd3r_constant` at α = 0.10, 0.15, 0.20, 0.25, 0.33, on TUM 1000f. Each is compared with the fresh `ortho_check` from E1b (0.05454).

**Check.** `const033` is compared with the logged older-code 0.06591. This is reported and does not gate anything. Later comparisons use the fresh value.

**Prediction P6b.** `ortho_check` is better than every one of the five flat multipliers.

| Outcome | Meaning | Consequence for the replies |
|---|---|---|
| Ortho better than all five | No tested flat step size reaches ortho on TUM | May say so, scoped to α in [0.10, 0.5] on TUM |
| Tied with any | Scale alone can come within noise of ortho on TUM | Must say so. "Direction helps on TUM" stays as the same-norm result, and we must not imply that scale cannot do the same |
| Any flat multiplier better | A tuned flat multiplier beats ortho on TUM | Must say so plainly, and the TUM case for direction narrows to the fixed-norm comparison |

**Selection.** The flat sweep is tuned on TUM, as ortho was, so this compares two configurations selected on the same data.

**Boundary.** If the lowest mean is at α=0.10, the result is reported as sitting on the edge of the sweep. Any extension needs a dated amendment.

**Multiplicity.** The binding rule is the common rule. Holm-adjusted p over the five tests is reported beside it.

## E6c — α∥=0.15

**Arms.** `ortho_a015` is the update of `ddd3r` with α∥=0.15 and logs the applied norm. `normreplay_a015` replays that log. Both run on TUM 1000f, ScanNet 1000f and ARKitScenes 500f. `ortho_a015` also runs on `kitti_odo_full`, with no replay.

**Check.** `ortho_a015` must match the logged `ddd3r_a15` within 2.2% on TUM (0.05504) and ScanNet (0.39886). This also tests the folder names, which have misled us before. A miss blocks every ruling in E6c until explained.

**Predictions.**

- **P6c1.** On TUM, `ortho_a015` is better than `normreplay_a015`.
- **P6c2.** On ScanNet, `normreplay_a015` is still better than `ortho_a015`, and the harm is smaller than at α∥=0.05. "Smaller" means the 95% paired bootstrap CI of `(ortho − replay at 0.05) − (ortho − replay at 0.15)` lies entirely above zero.
- **P6c3, out of sample.** On ARKitScenes, `ortho_a015` is better than `ortho_check`, and it is not worse than TTT3R or TTSA3R by the common rule.
- **P6c4, out of sample.** On KITTI, `ortho_a015` is better than `brake_t1` on r_rel. ATE RMSE is reported with no prediction.

| Outcome | Meaning | Consequence for the replies |
|---|---|---|
| P6c1 holds, ScanNet same-norm comparison tied | One setting helps on TUM and shows no measured harm on ScanNet at identical norm | May say so, with the CI, as a post hoc setting |
| P6c1 and P6c2 hold | The harm shrinks and does not vanish | Report it as a sensitivity result |
| P6c1 tied or reversed | At this setting the TUM gain is not credited to direction | Remove the claim that α∥=0.15 keeps the directional gain on TUM |
| ScanNet harm does not shrink | Relaxing α∥ changes the error but not direction's share of it | Remove the implication that the ScanNet loss belongs to the operating point |
| P6c3 fails | The ScanNet improvement does not carry to unseen data | Disclose it beside any mention of α∥=0.15 |
| P6c4 fails | The KITTI rotation gain belongs to the default setting | Disclose it beside any mention of α∥=0.15 |

## Budget

About 17 GPU-hours at the speeds measured in E1–E5, across three workers. E6a and E6b should finish first, in about 3 to 4 hours. If the queue is projected to end after 2026-11-12, stop and report rather than cutting arms silently, since the discussion closes on 2026-11-18.

---

## Results, 2026-11-07, E6a and E6b

Everything above the line is the registered text, unchanged. Numbers come from `results/e6.txt`.

**Incident before the queue.** A test written before `run_arm.sh` had a dry-run switch started a real `const033` on GPU 0 at 13:38, after the hash was recorded. It ran for about a minute and wrote no sequence and no ATE. It was stopped, its empty folders and log were removed, and the queue started clean at 13:57. The script now has a dry-run switch and creates folders only after it has checked the arm name.

### E6a

**Check, passed.** `ortho_check` matches `ortho_plain` with a 0.0000% gap on all 11 sequences, on ATE and on r_rel.

**Replay wiring, not a registered check.** There are 11 norm logs, each frames−1 × 768, all written by `ortho_check` before `normreplay` started. Running past a log would raise. Sequence 04 repeats the smoke run (ATE 11.46, r_rel 4.187, t_rel 9.96). Unit tests cover the per-token norm. The realized norm was not measured separately on KITTI.

**Ruling: P6a holds.** On r_rel, ortho is better than `normreplay`: 11.88 vs 16.78 deg/100m (−29.2%), wins 10/11, CI [−8.36, −1.75], p=0.0186. The row "Ortho better" applies, so we may say direction lowers segment rotation error on KITTI at identical norm.

| Beside it, no prediction | Ortho | Same norm | Wins | p | Rule |
|---|---|---|---|---|---|
| t_rel | 53.77 | 60.13 | 7/11 | 0.17 | tied |
| ATE RMSE | 160.98 | 180.51 | 6/11 | 0.32 | tied |
| RPE translation | 8.04 | 4.65 | 0/11 | 0.0010 | same norm better |
| RPE rotation | 5.94 | 4.86 | 5/11 | 0.52 | tied |

**Not registered.**
- Sequence 08 reverses (24.92 vs 19.46). Leaving out any one sequence keeps the verdict, with a largest p of 0.037.
- On r_rel, `normreplay` ties `brake_t1` (16.78 vs 17.66, p=0.17) and `const05` (vs 17.82, p=0.15), and is better than TTT3R (vs 20.60, wins 10/11, p=0.0068).

**Limits.**
- r_rel became the primary metric after the scalar comparison in E5b had been seen. The same-norm arm is new, but the choice of metric is not independent of the earlier look.
- r_rel comes from `eval/relpose/eval_kitti_odom.py` with Sim(3) alignment, and is not the devkit metric.
- α=0.5 is the only flat multiplier run on KITTI with this code, so nothing here covers other flat values.

### E6b

**Check, reported only.** `const033` gives 0.06483 against the older-code 0.06591, tied (p=0.95).

**Ruling: P6b fails.** The row "Tied with any" applies. Scale alone can come within noise of ortho on TUM, and we must say so. "Direction helps on TUM" stays as the same-norm result and must not imply that scale cannot do the same.

| α, against ortho 0.05454 | Flat mean | Ortho wins | CI of ortho − flat | p | Holm p | Rule |
|---|---|---|---|---|---|---|
| 0.10 | 0.08618 | 6/8 | [−0.0771, +0.0000] | 0.195 | 0.445 | tied |
| 0.15 | 0.08041 | 7/8 | [−0.0709, +0.0011] | 0.109 | 0.438 | tied |
| 0.20 | 0.07305 | 6/8 | [−0.0535, +0.0009] | 0.250 | 0.445 | tied |
| 0.25 | 0.06547 | 6/8 | [−0.0288, −0.0006] | 0.148 | 0.445 | tied |
| 0.33 | 0.06483 | 8/8 | [−0.0162, −0.0055] | 0.0078 | 0.039 | ortho better |

No flat multiplier is better than ortho, so the third row does not apply. The lowest flat mean is at α=0.33, so the boundary clause does not apply.

**Not registered.** One sequence, `sitting_halfsphere`, carries much of the gap for α≤0.25 (flat 0.13 to 0.24, ortho 0.062). Without it, α=0.20 gives 0.05498 and α=0.25 gives 0.05581 against ortho 0.05347.

### Effect on the earlier comparison with brake

Against the logged brake value, the fresh `const033` changes the E4 reading on TUM from tied to brake better: 0.06483 vs 0.06339 (+2.3%, const wins 2/8, CI [+0.0003, +0.0033], p=0.039). `normreplay` against `const033` changes from tied to `const033` better (0.07437 vs 0.06483, p=0.016). The reply to qpVA still prints the older-code tie for TUM and has to change. `brake_t1` on TUM is being rerun on the current code, outside the registered predictions, to make this comparison like for like. Its result goes in a later dated section.

## Results, 2026-11-07, brake rerun on TUM

`brake_t1` on TUM, run on the current code outside the registered predictions, reproduces the logged brake on all 8 sequences with a 0.0000% gap (mean 0.06339). So the comparison above is like for like. Brake is better than `const033` on TUM, 0.06339 vs 0.06483 (+2.3%, const wins 2/8, CI [+0.0003, +0.0033], p=0.039).

- The gap sits just above the 2.2% floor, and `walking_rpy` supplies about two thirds of it (0.1290 vs 0.1366).
- `const033` has the lower mean RPE, translation 0.01138 vs 0.01337 and rotation 0.680 vs 0.889. These are means only, with no paired test.
- `const033` closes about 90% of brake's gap to const 0.5 on TUM, since (0.07859 − 0.06483) / (0.07859 − 0.06339) = 0.905.

The reply to qpVA must print this instead of the older-code tie.

## Results, 2026-11-07, E6c

**Check, passed.** `ortho_a015` matches the logged values on TUM (0.05504) and ScanNet (0.39886), with a 0.000% gap on both.

**Rulings: all four predictions hold.** The row "P6c1 and P6c2 hold" applies, so α∥=0.15 is reported as a sensitivity result. The ScanNet harm shrinks and does not vanish. As registered, no outcome makes this setting the default.

| Prediction | Comparison | Means | Wins | CI | p | Result |
|---|---|---|---|---|---|---|
| P6c1, TUM | ortho vs same norm | 0.05504 vs 0.07253 | 7/8 | [−0.0293, −0.0071] | 0.016 | ortho better |
| P6c2, ScanNet | ortho vs same norm | 0.39886 vs 0.29964 | 13/65 | [+0.0679, +0.1364] | <0.001 | same norm better |
| P6c2, ScanNet | harm at 0.05 − harm at 0.15 | +0.0705 | | [+0.0312, +0.1163] | | smaller |
| P6c3, ARKitScenes | `ortho_a015` vs `ortho_check` | 0.56678 vs 0.60927 | 13/18 | [−0.0690, −0.0168] | 0.012 | `ortho_a015` better |
| P6c3, ARKitScenes | `ortho_a015` vs TTT3R | 0.56678 vs 0.55804 | 8/18 | [−0.0614, +0.0912] | 0.91 | tied |
| P6c3, ARKitScenes | `ortho_a015` vs TTSA3R | 0.56678 vs 0.55591 | 7/18 | [−0.0573, +0.0915] | 0.81 | tied |
| P6c4, KITTI r_rel | `ortho_a015` vs `brake_t1` | 15.44 vs 17.66 | 10/11 | [−3.69, −0.87] | 0.0098 | `ortho_a015` better |

ATE RMSE on KITTI, reported with no prediction, ties brake (172.78 vs 172.21, p=0.76).

**Not registered.**
- α∥=0.15 gives up part of the KITTI rotation gain. r_rel is 15.44 against the default's 11.88 (wins 1/11, p=0.024), and only sequence 08 improves. ATE RMSE ties the default (172.78 vs 160.98, p=0.41).
- On ScanNet, the same-norm replay at α∥=0.15 is still worse than brake (0.29964 vs 0.26147, p<0.001).
- On ARKitScenes, `ortho_a015` ties brake (0.56678 vs 0.53907, p=0.81) and its own same-norm replay (vs 0.54193, p=0.56), so this result gives direction no credit there.
- At α∥=0.15, RPE translation and rotation tie the same-norm replay on TUM, ScanNet and ARKitScenes.
