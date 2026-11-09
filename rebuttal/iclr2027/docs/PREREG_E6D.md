# E6d — flat multipliers on KITTI Odometry, predictions and decision rules

> **Fixed before any E6d arm produced a value.** Its SHA-256 is written to `results/e6d_registration.sha256`, and `scripts/run_e6d.sh` refuses to start if the file no longer matches. Nothing below may be edited after results exist. Corrections and results go in dated sections at the end.

Submission: ICLR 2027, 28147. Registered 2026-11-07. Follows `PREREG_E6.md`, whose common protocol and comparison rule apply unchanged.

## Why this exists

E6a found ortho better than its same-norm replay on KITTI r_rel. E6b found that flat multipliers from α=0.10 to 0.25 tie ortho on TUM. On KITTI, α=0.5 is the only flat multiplier with a segment metric, so the replies cannot yet say whether a smaller flat step reaches ortho's r_rel.

| Statement under test | Current evidence |
|---|---|
| No flat step size reaches ortho's segment rotation error on KITTI | Ortho beats α=0.5 (10/11), and no other flat value has an r_rel |

## What was already known at registration

This registration is not blind to the reference arms. These values were read before writing it.

| KITTI Odometry, 11 sequences | r_rel (deg/100m) | ATE RMSE |
|---|---|---|
| TTT3R | 20.603 | 180.23 |
| Const 0.5 | 17.817 | 178.24 |
| Brake τ=1 | 17.661 | 172.21 |
| Same norm | 16.780 | 180.51 |
| Ortho | 11.881 | 160.98 |

**α=0.33 on two sequences.** A diagnostic run gives ATE 60.68 on sequence 03 and 5.10 on sequence 04, against ortho's 39.91 and 17.79. It has RPEs but no r_rel, since none was computed. It is also the check below.

**E6b was read first.** The same statement failed on TUM, so P6d is the statement the replies would like to make, not a forecast with strong support.

**Nothing exists for:** r_rel at any flat value other than α=0.5, or any KITTI result at α=0.10, 0.15, 0.20 or 0.25.

## Protocol

The E6 common protocol applies. `git diff 43f7e29 -- src eval` is empty at registration, so the model code matches E1–E6. All 11 sequences run, and no sequence is excluded after the fact. Every outcome is reported, whichever way it goes.

**Arms.** `const010`, `const015`, `const020`, `const025`, `const033`, which are `ddd3r_constant` at α = 0.10, 0.15, 0.20, 0.25, 0.33, on `kitti_odo_full`. Each is compared with the fresh `ortho_check` from E6a. `const05` from E5b completes the sweep and is not rerun.

**Check.** `const033` must match the diagnostic ATE within 0.5% on sequence 03 (60.68427) and sequence 04 (5.09581). This also tests that `--alpha` reaches the model on KITTI. A miss blocks the ruling until explained.

**Primary metric.** r_rel from `eval/relpose/eval_kitti_odom.py`, the same script and metric as E6a. It aligns with Sim(3) and averages per-sequence means, so it is not the KITTI devkit metric and must not be called official. ATE RMSE, t_rel and both RPEs are reported beside it, with no prediction.

## Prediction and consequences

**Prediction P6d.** `ortho_check` is better than every one of the five flat multipliers on r_rel.

| Outcome on r_rel | Meaning | Consequence for the replies |
|---|---|---|
| Ortho better than all five | No tested flat step size reaches ortho's r_rel on KITTI | May say so, scoped to α in [0.10, 0.5] on KITTI Odometry |
| Tied with any | Scale alone can come within noise of ortho's r_rel on KITTI | Must say so. The identical-norm sentence from E6a stays, and must not imply that only decomposition reaches this r_rel |
| Any flat multiplier better | A flat multiplier beats ortho's r_rel on KITTI | Must say so plainly, and the KITTI case for direction narrows to the fixed-norm comparison |

**Selection.** None of the five values was tuned on KITTI. They repeat E6b for symmetry. Ortho has to beat all five, which gives the flat side the advantage of a best-of-five reading.

**Boundary.** If the lowest r_rel mean is at α=0.10, the result is reported as sitting on the edge of the sweep. Any extension needs a dated amendment.

**Multiplicity.** The binding rule is the common rule. Holm-adjusted p over the five tests is reported beside it.

## Budget

About 1.1 to 1.5 hours per arm while other jobs share the GPU, so 5.5 to 7.5 GPU-hours, in two workers on GPU 1. If the queue is projected to end after 2026-11-12, stop and report rather than cutting arms silently, since the discussion closes on 2026-11-18.

---

## Results, 2026-11-07

Everything above the line is the registered text, unchanged. Numbers come from `results/e6.txt`.

**Check, passed.** `const033` matches the diagnostic ATE on sequences 03 and 04 with a 0.000% gap.

**Ruling: P6d fails.** The row "Tied with any" applies. Scale alone can come within noise of ortho's r_rel on KITTI, and we must say so. The identical-norm sentence from E6a stays and must not imply that only decomposition reaches this r_rel.

| α, against ortho 11.88 | Flat r_rel | Ortho wins | CI of ortho − flat | p | Holm p | Rule |
|---|---|---|---|---|---|---|
| 0.10 | 12.41 | 5/11 | [−5.09, +4.07] | 0.97 | 1.00 | tied |
| 0.15 | 13.92 | 7/11 | [−6.05, +1.54] | 0.52 | 1.00 | tied |
| 0.20 | 16.15 | 8/11 | [−7.74, −1.10] | 0.054 | 0.16 | tied |
| 0.25 | 17.18 | 10/11 | [−8.86, −2.27] | 0.0068 | 0.027 | ortho better |
| 0.33 | 17.61 | 10/11 | [−9.24, −2.95] | 0.0020 | 0.0098 | ortho better |
| 0.5, from E5b | 17.82 | 10/11 | [−10.09, −2.30] | 0.019 | not in the Holm set | ortho better |

No flat multiplier is better than ortho, so the third row does not apply.

**Boundary applies.** The lowest r_rel mean is at α=0.10, the edge of the sweep. r_rel falls at every step from α=0.5 to α=0.10, so a smaller value could go lower. Nothing below 0.10 was run, and any extension needs a dated amendment.

**Beside it, no prediction.**

| Against ortho | α=0.10 | α=0.15 | α=0.20 | α=0.25 | α=0.33 | Ortho |
|---|---|---|---|---|---|---|
| t_rel | 52.52 | 52.80 | 56.40 | 58.04 | 62.85 | 53.77 |
| ATE RMSE | 178.79 | 187.88 | 189.08 | 189.74 | 175.87 | 160.98 |
| RPE translation | 11.70 | 5.86 | 5.01 | 4.52 | 5.14 | 8.04 |
| RPE rotation | 6.84 | 7.16 | 7.08 | 5.83 | 6.54 | 5.94 |

- t_rel, ATE RMSE and RPE rotation tie ortho at all five values, by the common rule.
- On RPE translation, ortho is better than α=0.10 (wins 10/11, p=0.032), ties α=0.15, and is worse than α=0.20, 0.25 and 0.33 (p ≤ 0.002).

**Not registered.** α=0.10 and ortho reach similar means through different sequences. α=0.10 is lower on 03 (6.79 vs 17.18) and 08 (11.37 vs 24.92), and ortho is lower on 07 (14.15 vs 29.47) and 09 (8.78 vs 17.15).
