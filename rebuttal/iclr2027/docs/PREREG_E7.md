# E7 — direction at each scale, predictions and decision rules

> **Fixed before any E7 arm produced a value.** Its SHA-256 is written to `results/e7_registration.sha256`, and `scripts/run_e7.sh` refuses to start if the file no longer matches. Nothing below may be edited after results exist. Corrections and results go in dated sections at the end.

Submission: ICLR 2027, 28147. Registered 2026-11-07. Follows `PREREG_E6.md` and `PREREG_E6D.md`, whose comparison rule applies unchanged.

## Why this exists

E6a found that direction lowers KITTI r_rel at ortho's own norm. E6b and E6d found that flat multipliers from α=0.10 to 0.20 tie ortho on TUM and on KITTI r_rel. Those flat values take much smaller steps than ortho, whose mean gain is 0.34 to 0.37. So it is open whether direction and scale substitute for each other or add up. The replay tested direction at one norm only.

| Statement under test | Current evidence |
|---|---|
| Direction adds a gain on top of scale, at each scale | Tested only at ortho's norm schedule |
| The ScanNet harm of direction shrinks as the step shrinks | Shown for a larger α∥, not for a smaller step |
| No single flat multiplier is at least as good as ortho everywhere | α=0.10 has not been run on ScanNet or ARKitScenes |

## What was already known at registration

This registration is not blind to the reference arms. These values were read before writing it.

| Primary metric | const 0.10 | const 0.20 | const 0.33 | const 0.5 | Same norm | Brake τ=1 | Ortho |
|---|---|---|---|---|---|---|---|
| TUM ATE RMSE | 0.08618 | 0.07305 | 0.06483 | 0.07859 | 0.07437 | 0.06339 | 0.05454 |
| KITTI r_rel (deg/100m) | 12.41 | 16.15 | 17.61 | 17.82 | 16.78 | 17.66 | 11.88 |
| ScanNet ATE RMSE | not run | not run | 0.26237 | 0.28274 | 0.31858 | 0.26147 | 0.48829 |
| ARKitScenes ATE RMSE | not run | not run | not run | 0.52754 | 0.54458 | 0.53907 | 0.60927 |

Ortho's mean gain is 0.3577 on TUM, 0.3437 on KITTI, 0.3558 on ScanNet and 0.3743 on ARKitScenes.

Post hoc comparisons already read: on TUM, ortho has better RPE rotation than α=0.10, 0.15 and 0.20 (p ≤ 0.016), and ties it against α=0.25 and 0.33. On KITTI, ortho has worse RPE translation than α=0.20, 0.25 and 0.33.

**Why these scales.** They were chosen after reading the sweeps above. α=0.33 is the flat value closest to ortho's mean gain. α=0.10 is the smallest value run, and has the lowest KITTI r_rel. α=0.20 lies between them. The direction keeps the default ratio α∥/α⊥ = 0.1. **No outcome below makes any of these arms the paper's default.**

**Nothing exists for:** any `dir` arm, const 0.10 on ScanNet, or const 0.10, 0.20 or 0.33 on ARKitScenes.

## Protocol

**Code.** `--ortho_scale` is added to `eval/relpose/launch.py` and `_delta_ortho_update`, on top of commit 43f7e29. The SHA-256 of `git diff 43f7e29 -- src eval` is written to `results/e7_code.sha256`. Without the flag the update path is unchanged. Unit tests are in `tests/test_update_rules.py`.

**Arm definition.** `dirNNN` updates each token by α · ‖δ‖ · unit(α⊥δ⊥ + α∥δ∥), with α⊥=0.5, α∥=0.05, β_ema=0.95, γ=0, and the TTT3R gate applied as usual. The first frame moves by α·δ. Its applied norm is therefore gate · α · ‖δ‖, the same rule as `ddd3r_constant` at α. The rule matches, but the logged values do not, since each run has its own trajectory, so this compares the closed-loop effect of the decomposition at a matching norm rule.

**Comparison rule.** From `scripts/stats_rule.py`. "A is better than B" needs a lower mean, p < 0.05 and a gap above 2.2%. Anything else is "tied", and the CI is reported beside it. ScanNet runs on the same 65 valid scenes as E1b. No sequence is excluded after the fact. Every outcome is reported, whichever way it goes.

**Arms.**

| Dataset | New `dir` arms | New flat arms | Flat arms reused |
|---|---|---|---|
| TUM 1000f | 0.10, 0.20, 0.33 | none | E6b |
| KITTI Odometry | 0.10, 0.20, 0.33 | none | E6d |
| ARKitScenes 500f | 0.10, 0.20, 0.33 | 0.10, 0.20, 0.33 | none |
| ScanNet 1000f | 0.10, 0.33 | 0.10 | const 0.33 from E4 |

α=0.20 is not run on ScanNet, to save time.

**Checks.**

1. **Regression.** With the new code, `ortho_check` and `const033` on KITTI sequence 04 must match 17.78537 and 5.09581 within 0.5%. A miss stops the queue from starting.
2. **Isotropic limit.** `iso033`, which sets α∥=α⊥ and scale 0.33, must match `const033` on KITTI sequence 04 (5.09581) within 0.5%. A miss stops the queue from starting.
3. **Gain.** The logged gain of each `dirNNN` must equal its scale within 0.5% on every sequence. A miss blocks that arm's ruling until explained.

**Primary metrics.** ATE RMSE on TUM, ScanNet and ARKitScenes. r_rel on KITTI, from `eval/relpose/eval_kitti_odom.py`, which aligns with Sim(3) and is not the KITTI devkit metric. Both RPEs everywhere, and ATE RMSE and t_rel on KITTI, are reported beside them with no prediction. Each `dir` arm against the default ortho is also reported with no prediction.

## Predictions

- **P7a.** On TUM, `dirNNN` is better than `constNNN` at each of α = 0.10, 0.20, 0.33.
- **P7b.** On KITTI r_rel, `dirNNN` is better than `constNNN` at each of α = 0.10, 0.20, 0.33.
- **P7c.** On ScanNet, `const033` is better than `dir033`, and the harm is smaller at α=0.10. "Smaller" means the 95% paired bootstrap CI of `(dir033 − const033) − (dir010 − const010)` lies entirely above zero.
- **ARKitScenes.** No prediction. Reported in full.

These are the statements the replies would like to make. Support is strongest at α=0.33, and weak at α=0.10.

## Consequences for the replies

**TUM and KITTI r_rel, read for each dataset.**

| Outcome | Meaning | Consequence |
|---|---|---|
| `dir` better at all three scales on both datasets | Direction adds to scale at every tested norm | May say so, scoped to α in {0.10, 0.20, 0.33} on TUM and KITTI r_rel |
| `dir` better at α=0.33 but not at α=0.10 | Direction and smaller steps substitute on that dataset | Must say the gain fades as the step shrinks |
| Tied at all three on a dataset | No gain from direction at a flat norm there | Must say so, and scope the earlier identical-norm sentence to ortho's own norm schedule |
| Any `const` better than `dir` | Direction costs accuracy at that scale | Must say so plainly |
| Any other pattern | No simple account | Report each cell and draw no general statement |

**ScanNet.**

| Outcome | Meaning | Consequence |
|---|---|---|
| P7c holds | The harm shrinks with the step and does not vanish | May say so |
| P7c fails | The harm is not shown to depend on the step | Remove any implication that it does |
| `dir010` better than `const010` | Not predicted | Report it as not predicted |

**Flat multipliers.**

| Outcome | Meaning | Consequence |
|---|---|---|
| const 0.10 is not worse than ortho on TUM, KITTI r_rel, ScanNet and ARKitScenes | One flat value does not lose to ortho on any dataset tested | Must say so plainly in any reply that discusses direction |

α=0.20 cannot be checked this way, since it is not run on ScanNet.

**Multiplicity.** The binding rule is the common rule. Holm-adjusted p over the tests within each dataset is reported beside it.

**Default.** Any `dir` arm better than the default ortho is reported as a sensitivity result. It does not change the default.

## Budget

About 19 to 25 GPU-hours, in four workers across two GPUs, so about 7 to 9 hours of wall clock. If the queue is projected to end after 2026-11-10, stop and report rather than cutting arms silently, since the revised PDF needs the remaining time.

---

## Amendment 1, 2026-11-07, before any full E7 arm ran

Everything above the line is the registered text, unchanged. This amendment was written after the smoke runs and before any full `dir` arm or any perturbed arm started. Its file hash is in `results/e7_amendment1.sha256`.

### Check results

- **Check 1 passed.** `ortho_check` gives 17.78537 and `const033` gives 5.09581 on KITTI sequence 04, a 0.0000% gap for both.
- **Check 3, on the smoke arms.** `dir033` logs a gain of exactly 0.33 at every frame on KITTI sequence 04 and on TUM `walking_static`.
- **Check 2 missed.** `iso033` gives 5.01866 against 5.09581, a 1.51% gap against a 0.5% tolerance. As registered, this stopped the queue.

### Root cause of the miss

The check assumed that two numerically different but mathematically equal paths would agree within 0.5%. They do not, because a single sequence's ATE is sensitive to rounding-level perturbations. On KITTI sequence 04:

| Run | ATE | RPE translation |
|---|---|---|
| `ddd3r_constant`, α=0.33, rerun | 5.09581 | 1.77483 |
| `ddd3r_constant`, α=0.3300001 | 4.72539 | 1.85054 |
| `ddd3r_constant`, α=0.3299999 | 4.95570 | 1.93722 |
| `ddd3r`, α⊥=α∥=0.33, no new flag | 4.92276 | 1.81776 |
| `iso033`, the new flag | 5.01866 | 1.72505 |

The exact rerun matches bit for bit, so the code is deterministic. A change of about 3e-7 in α moves the ATE by up to 7.3%, so the 1.51% is inside what rounding produces. The older path with equal α also differs, by 3.4%, with no new code involved.

Position gap to `const033`, frame by frame:

| Run | Frame 5 | Frame 10 | Frame 20 | Frame 40 | Frame 80 | Frame 268 |
|---|---|---|---|---|---|---|
| `iso033` | 0.0003 | 0.0008 | 0.0113 | 0.0298 | 0.8137 | 5.8867 |
| α=0.3300001 | 0.0003 | 0.0012 | 0.0253 | 0.1933 | 0.1414 | 5.1252 |
| α=0.3299999 | 0.0004 | 0.0009 | 0.0120 | 0.0710 | 0.2704 | 4.4100 |
| `dir033`, a genuinely different arm | 0.1557 | 0.1046 | 1.3096 | 5.2969 | 6.6683 | 38.8480 |
| Ortho, a genuinely different arm | 0.0522 | 0.4286 | 0.2637 | 1.2669 | 5.7158 | 109.1886 |

`iso033` separates from `const033` no faster than a 1e-7 perturbation does, and much more slowly than arms that truly differ. The unit test `test_equal_alphas_reduce_to_constant_dampening` covers the formula. So the new path is equivalent, and the miss came from the tolerance.

### Deviation

The registered text says a miss stops the queue from starting. The queue starts anyway, with this explanation on record, because the check's purpose was to catch a code error and the frame-by-frame test above does that better. The 0.5% tolerance on this check is withdrawn.

### Perturbed arms, added because of this finding

The 2.2% floor is the largest gap seen between code versions of one configuration. It was never measured against rounding, and exact reruns say nothing about it. Two new arms measure that on the datasets carrying the claims for direction:

- `const033p`, `ddd3r_constant` at α=0.3300001.
- `orthop`, the default ortho at α⊥=0.5000001.

Both run on TUM 1000f and `kitti_odo_full`. They are compared with `const033` and `ortho_check`, on ATE RMSE for TUM and on r_rel and ATE RMSE for KITTI. The key comparisons are then repeated with the perturbed arm in place of the original: ortho against const 0.33 and const 0.10 on both datasets, and ortho against same norm on KITTI r_rel and TUM.

**Consequences, fixed now, before any perturbed result exists.**

| Outcome | Consequence |
|---|---|
| A perturbed arm's mean differs from the original by more than 2.2% on any primary metric | The replies must stop calling 2.2% our largest gap, and must print the measured gap. Any reported "better" whose gap lies below the measured gap must be described as within rounding noise |
| A verdict among the key comparisons changes under perturbation | The replies must report that verdict as not robust, or drop it |
| Both perturbed arms stay within 2.2% and no key verdict changes | The rule stands, and the replies may say the verdicts survive a rounding perturbation |

ScanNet and ARKitScenes get no perturbed arms, so nothing here measures rounding noise on them. These two arms are one draw each, so they give a lower bound on the noise and not its size.

## Results, 2026-11-08, rounding noise

Numbers come from `results/e7.txt`.

| Metric | `orthop` vs ortho | `const033p` vs const 0.33 | Largest single-sequence change |
|---|---|---|---|
| TUM ATE RMSE | 0.05613 vs 0.05454, +2.9% | 0.06786 vs 0.06483, +4.7% | 33.2% |
| KITTI r_rel | 10.29 vs 11.88, −13.4% | 17.15 vs 17.61, −2.6% | 40.4% |
| KITTI ATE RMSE | 172.16 vs 160.98, +6.9% | 179.09 vs 175.87, +1.8% | 78.4% |

**Ruling: the first row applies.** Both TUM gaps and the KITTI `orthop` gaps exceed 2.2%, so the replies must stop calling 2.2% our largest gap and must print the measured gaps. On KITTI r_rel, the common rule even calls `orthop` better than ortho (wins 9/11, p=0.0137), though the two differ only by rounding. So p<0.05 with a gap above 2.2% does not separate real effects from rounding on that metric.

**Reported "better" verdicts below the measured gap.** These must be described as within rounding noise:

- TUM, brake against const 0.33, a 2.3% gap against a measured 4.7%. It is the only such verdict on TUM.
- KITTI r_rel, α∥=0.15 against brake, a 12.6% gap against a measured 13.4%. This is P6c4. The prediction holds by the common rule, but the gap is inside the noise.

**Key verdicts: none changed.**

| Comparison | Original | With `orthop` |
|---|---|---|
| TUM, ortho vs const 0.33 | better, 8/8, −15.9% | better, 8/8, −13.4% |
| TUM, ortho vs const 0.10 | tied, p=0.20 | tied, p=0.20 |
| TUM, ortho vs same norm | better, 8/8, −26.7% | better, 8/8, −24.5% |
| KITTI r_rel, ortho vs const 0.33 | better, 10/11, −32.5% | better, 11/11, −41.5% |
| KITTI r_rel, ortho vs const 0.10 | tied, p=0.97 | tied, p=0.41 |
| KITTI r_rel, ortho vs same norm | better, 10/11, −29.2% | better, 10/11, −38.7% |

The gaps for the "better" rows all exceed the measured gaps. `orthop` against `const033p` is also better on both, 8/8 and 11/11. This is one draw, the same-norm arm was not perturbed, and ScanNet and ARKitScenes were not measured.

## Validity of r_rel, 2026-11-08, not registered

r_rel fell and RPE translation rose together across the arms, so we checked whether r_rel rewards a method for not turning. `scripts/kitti_metric_validity.py` scores a trajectory with ground-truth positions and its orientation frozen at the first frame. Output is in `results/kitti_metric_validity.txt`.

- The never-turning trajectory gets a mean r_rel of 21.87 deg/100m. That is close to CUT3R (22.26) and TTT3R (20.60), and far worse than ortho (11.88) or `dir033` (9.13).
- Every arm's predicted rotation over a 10-frame window is 1.5 to 1.8 times the ground truth's, so none under-rotates.

So r_rel is not rewarding under-rotation here. The check leaves two things open. The baselines' r_rel is about as poor as never turning at all, which limits what a gain over them means. And why r_rel falls while frame-level translation worsens has not been explained.

## Results, 2026-11-08, E7

The queue finished at 2026-11-08 05:08:29, before the 2026-11-10 limit, after about 5.3 hours of wall clock on two GPUs shared by four workers. Numbers come from `results/e7.txt` and `results/e7_kitti_metrics.txt`. Check 3 passes for every `dir` arm on all four datasets, with a largest gap of 0.0000%. Sequence counts are 8, 11, 65 and 18, as expected.

### P7a, TUM ATE RMSE: fails, tied at all three scales

| α | `dir` | `const` | `dir` wins | 95% CI | p | Verdict |
|---|---|---|---|---|---|---|
| 0.10 | 0.08489 | 0.08618 | 5/8 | [-0.0047, +0.0025] | 0.547 | tied |
| 0.20 | 0.08040 | 0.07305 | 5/8 | [-0.0043, +0.0243] | 0.844 | tied |
| 0.33 | 0.05924 | 0.06483 | 5/8 | [-0.0158, +0.0054] | 0.313 | tied |

Holm-adjusted p is 1.0, 1.0 and 0.94. Every `dir` arm also ties ortho (p ≥ 0.31). Beside it, `dir010` has worse RPE rotation than `const010` (2.21 vs 1.75, wins 1/8, p=0.0156). The other RPE comparisons tie.

**Ruling.** The row "Tied at all three on a dataset" applies. The replies must say that direction at a flat norm gives no gain on TUM, and must scope the identical-norm sentence to ortho's own norm schedule.

### P7b, KITTI r_rel: holds at all three scales

| α | `dir` | `const` | `dir` wins | 95% CI | p | Gap |
|---|---|---|---|---|---|---|
| 0.10 | 7.68 | 12.41 | 8/11 | [-7.55, -2.07] | 0.0137 | −38.1% |
| 0.20 | 7.88 | 16.15 | 9/11 | [-11.13, -5.23] | 0.0049 | −51.2% |
| 0.33 | 9.13 | 17.61 | 11/11 | [-11.27, -5.72] | 0.0010 | −48.2% |

Holm-adjusted p is 0.0137, 0.0098 and 0.0029. Every gap exceeds the 13.4% measured on this metric under Amendment 1. Every `dir` arm ties ortho on r_rel (p = 0.054, 0.054, 0.175).

Beside it, with no prediction:

| α | ATE RMSE | t_rel | RPE translation | RPE rotation |
|---|---|---|---|---|
| 0.10 | worse, 208.09 vs 178.79, 1/11, p=0.0049 | tied, p=0.054 | worse, 14.93 vs 11.70, 2/11, p=0.042 | tied |
| 0.20 | tied, p=0.90 | tied, p=0.70 | worse, 14.44 vs 5.01, 0/11, p=0.0010 | tied |
| 0.33 | tied, p=0.28 | better, 51.94 vs 62.85, 8/11, p=0.042 | tied, 8.60 vs 5.14, 1/11, p=0.054 | tied |

The t_rel gain at 0.33 is 17.4%. A rounding perturbation of ortho moved t_rel by 4.1%, measured after the fact from the Amendment 1 arms.

**Ruling.** On KITTI `dir` is better at all three scales, but TUM is tied, so the first row, "on both datasets", is not met. No other row describes the KITTI pattern, so "Any other pattern" applies. The replies may report each KITTI cell, with its costs beside it and the TUM ties next to it. They may not state, as a general claim, that direction adds to scale or that the two combine. An informal reading given to the author before this ruling, that the two "can stack on this metric", went beyond the table and is withdrawn.

### P7c, ScanNet ATE RMSE: fails

`const033` is better than `dir033` (0.26237 vs 0.51739, `dir` wins 3/65, p<0.001), so the first half holds. `const010` is also better than `dir010` (0.38189 vs 0.63510, 5/65, p<0.001). The harm at 0.33 minus the harm at 0.10 is +0.0018, CI [-0.0734, +0.0677], which does not lie above zero. So the second half fails.

**Ruling.** The row "P7c fails" applies. Remove any implication that the harm depends on the step. The row "`dir010` better than `const010`" does not apply.

Against ortho, with no prediction: `dir010` is worse (0.635 vs 0.488, 20/65, p<0.001), and `dir033` is worse (0.517 vs 0.488, 26/65, p=0.046), a 6.0% gap. Rounding was not measured on ScanNet, and 6.0% is below the 6.9% measured on KITTI ATE, so the second verdict is not taken as robust. Both RPEs are worse for `dir` at both scales.

### ARKitScenes ATE RMSE, no prediction

| α | `dir` | `const` | `dir` wins | 95% CI | p | Verdict |
|---|---|---|---|---|---|---|
| 0.10 | 0.69095 | 0.60956 | 2/18 | [+0.0421, +0.1327] | 0.0003 | const better |
| 0.20 | 0.64603 | 0.56085 | 3/18 | [+0.0377, +0.1460] | 0.0025 | const better |
| 0.33 | 0.60599 | 0.54051 | 6/18 | [+0.0008, +0.1441] | 0.085 | tied |

Holm-adjusted p is 0.0010, 0.0049 and 0.085. Rounding was not measured on ARKitScenes. `dir010` is worse than ortho (4/18, p=0.020). `dir020` and `dir033` tie it. RPEs tie at every scale.

**Ruling.** No row binds ARKitScenes. Every outcome is reported, so any reply that cites the KITTI cells also cites these, and the KITTI cells never stand alone.

Post hoc, not registered: `const033` on ARKitScenes ties brake (0.54051 vs 0.53907, p=0.78), TTT3R (p=0.35), const 0.5 (p=0.47) and ortho (p=0.12).

### Flat multipliers against ortho

| Primary metric | ortho vs const 0.10 | ortho vs const 0.33 |
|---|---|---|
| TUM ATE | tied, p=0.20 | ortho better, 8/8 |
| KITTI r_rel | tied, p=0.97 | ortho better, 10/11 |
| ScanNet ATE | const 0.10 better, 0.382 vs 0.488, ortho wins 17/65, p<0.001 | const 0.33 better, 3/65 |
| ARKitScenes ATE | tied, p=0.50 | tied, p=0.12 |

**Ruling.** The flat row applies to const 0.10. Any reply that discusses direction must say plainly that const 0.10 is not worse than ortho on any of the four datasets.

Post hoc, not registered: ortho has lower RPE rotation than const 0.10 on TUM (8/8, p=0.0078), ScanNet (51/65, p=0.0002) and ARKitScenes (17/18, p=0.0003), and ties it on KITTI. At a matched flat step, direction does not lower RPE rotation (worse on TUM at 0.10, worse on ScanNet at both scales, tied on ARKitScenes). So this advantage goes with ortho's larger step and not with direction. On KITTI, ATE and t_rel also tie between ortho and const 0.10. On ScanNet, const 0.10 is worse than const 0.33 (15/65, p<0.001).

### Default

No `dir` arm is better than ortho on any primary metric, so no sensitivity result arises and the default is unchanged.

### Summary of the cells

At a matched flat step, direction lowers KITTI r_rel at every scale tested, above the measured rounding gap, at a cost in ATE at 0.10 and in RPE translation at 0.10 and 0.20. It ties on TUM. It raises ATE on ScanNet at both scales and on ARKitScenes at 0.10 and 0.20. One flat multiplier, 0.10, is not worse than ortho on any dataset's primary metric.
