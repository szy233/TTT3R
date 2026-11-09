# E1–E5 — predictions and decision rules

> **Committed before any of the new arms produced an ATE.** Check `git log` for this file against the creation time of `rebuttal/iclr2027/results/relpose/`. Nothing below may be edited after results exist. Corrections and results go in dated sections at the end.

Submission: ICLR 2027, 28147. Registered 2026-11-06. Issue IDs refer to `../ISSUE_BOARD.md`.

## What was already known at registration

This registration is not blind to the reference arms. These logs exist and were read before writing it:

| Dataset (1000f unless noted) | TTT3R | const 0.5 | const 0.33 | brake τ=1 | ortho |
|---|---|---|---|---|---|
| TUM | 0.1025 | 0.0786 | 0.0659 | 0.0634 | 0.0545 |
| ScanNet | 0.4058 | 0.2827 | not run | 0.2615 | 0.4883 |
| ARKitScenes 500f | 0.558 | 0.528 | not run | 0.539 | 0.609 |

No ATE exists for any of: norm-matched update, projection onto the previous direction, frozen gates, ScanNet const 0.33, KITTI brake τ=1 under the `kitti_odo_full` protocol.

## Common protocol

- Checkpoint `model/cut3r_512_dpt_4_64.pth`, `--size 512`, code at the commit that adds this file.
- Metric: per-sequence ATE RMSE after Sim(3), read from `_error_log.txt`, compared over the sequences valid in both arms.
- Statistics: two-sided Wilcoxon signed-rank (exact for n ≤ 16) and a paired bootstrap 95% CI of the mean difference (20000 resamples, seed 0).
- **"A is better than B"** requires all three: lower mean ATE, p < 0.05, and a relative gap above 2.2%. The 2.2% is the largest gap seen between two nominally identical configurations (`results/paired_stats.txt`).
- **"Tied"** is everything else. A tie is not evidence of equivalence, so the CI is always reported beside it.

## Instrument check (runs first)

Rerun `ddd3r` on TUM 1000f with this code and compare with the logged mean of 0.05454.

- Within 2.2%: logged reference arms are used for ScanNet and ARKitScenes.
- Outside 2.2%: stop, find the cause, and rerun every reference arm before any ruling.

## E1 — same norm, no directional decomposition

**Answers** R2-Q1, R2-C1, R1-C3, R3-C1.

**Arm `normmatch`.** Identical to `ddd3r` (α⊥=0.5, α∥=0.05, β_ema=0.95, γ=0), except that the update is `S + β_t · g_t · δ_t` with `g_t = ‖α⊥δ⊥ + α∥δ∥‖ / ‖δ‖` per token.

**Data.** TUM 1000f (8), ScanNet 1000f (65 valid), ARKitScenes 500f (18).

**Manipulation check.** The run is closed-loop, so the state diverges from the `ddd3r` run and the norm match holds as a rule, not as a replay. Compare the mean gain of `ddd3r` and `normmatch` on TUM, and on the first five valid ScanNet scenes (scene0707_00, scene0709_00, scene0710_00, scene0712_00, scene0714_00). A gap above 10% is reported beside the verdict, because the arm would then not answer "same norm" as asked.

**Predictions.**

- **P1.** On TUM, `ddd3r` is better than `normmatch`.
- **P2a.** On ScanNet, `normmatch` is better than `ddd3r`.
- **P2b.** On ARKitScenes, `normmatch` is better than `ddd3r`.

**What each outcome means.**

| Outcome | Meaning | Consequence for the paper |
|---|---|---|
| P1, P2a, P2b all hold | Direction has an independent effect, and its sign follows drift energy | Keep the directional claim, scoped to low-drift scenes |
| P1 tied | TUM's gain comes from per-token magnitude, not direction | Say so plainly. The contribution becomes diagnosis and magnitude regulation, and direction becomes a tested hypothesis |
| P1 reversed (`normmatch` better on TUM) | Decomposition has no benefit even in its best setting | As above, and recommend `normmatch` over `ddd3r` |
| P1 holds, P2a or P2b fails | Direction helps on TUM, and the high-drift loss is not explained by direction alone | Report the failure. Drop the claim that drift energy sets the sign of the directional effect |

## E2 — projection timing

**Answers** R3-Q2a.

**Arm `prevdir`.** Identical to `ddd3r`, except that the update is decomposed against `d_{t-1}`, before the EMA absorbs the current delta. The tracker itself advances as usual.

**Data.** TUM 1000f, ScanNet 1000f. **No directional prediction.** The comparison with `ddd3r` is reported as it comes.

## E3 — token-level temporal adaptation

**Answers** R1-C4.

**(a) Measurement.** For TTT3R and TTSA3R, log β_{t,k} and split its variance into token mean, frame mean and residual (the token-level temporal part). Report the mean per-token temporal standard deviation next to the frame-averaged one the paper reports.

**(b) Functional test.** Arms `ttt3r_frozen` and `ttsa3r_frozen` replace β_{t,k} with that token's time mean from the logged run of the same sequence. This keeps spatial selectivity and removes all temporal adaptation.

**Data.** TUM 1000f (8) and the first 20 valid ScanNet scenes sorted by name: scene0707_00 scene0709_00 scene0710_00 scene0712_00 scene0714_00 scene0715_00 scene0717_00 scene0718_00 scene0719_00 scene0720_00 scene0721_00 scene0722_00 scene0723_00 scene0724_00 scene0725_00 scene0726_00 scene0731_00 scene0732_00 scene0733_00 scene0734_00.

**Predictions.** P3a (TTT3R) and P3b (TTSA3R): freezing does not hurt.

| Verdict, per method and dataset | Criterion |
|---|---|
| Supported | Upper CI bound of (frozen − live) is at most 5% of the live mean |
| Refuted | Frozen is worse by the common rule (lower mean for live, p < 0.05, gap above 2.2%) |
| Inconclusive | Anything else |

**Consequence.** If refuted, withdraw "provides no real temporal adaptivity" for that method and narrow the diagnosis to the frame-averaged level. If inconclusive, the paper states the measurement only and claims nothing about usefulness.

## E4 — brake against a constant of the same scale

**Answers** S5, supports R1-C3.

**Arm `const033`.** `ddd3r_constant` with α=0.33 on ScanNet 1000f.

**Prediction P4.** `ddd3r_brake` (τ=1) and `const033` are tied on ScanNet, as they already are on TUM (p=0.55).

**Consequence.** If tied, brake's gain over const 0.5 is a scale effect and the paper must not credit it to temporal adaptivity. If brake is better, brake carries something beyond scale on ScanNet.

## E5 — KITTI under one temperature and one protocol

**Answers** S1, S3. **No directional prediction**, since the report's brake result is τ=2 and τ=1 has never been run under this protocol.

**Check first.** Rebuild `kitti_odo_full` on this machine with the report's preparation script, then run sequence 04.

| Arm | Report value (ATE RMSE) | Tolerance |
|---|---|---|
| CUT3R | 32.45 | 5% |
| brake τ=2 | 6.66 | 5% |

The 5% allows for the different GPU (the report used an H200).

- **Pass:** run brake τ=1 on all 11 sequences and report ATE mean and RMSE beside the report's values.
- **Fail:** stop E5 and report to the author. Rerunning all five methods here is about 3.3 GPU-hours beyond the approved budget and needs a separate go-ahead.

**Table 5 provenance.** Run `kitti_odom` sequence 04 at τ=1 and τ=2. The temperature that reproduces the logged 7.05 within 1% identifies what Table 5 used. If neither does, Table 5's KITTI columns are treated as unverifiable.

## Budget and reporting

About 12.5 GPU-hours at the nominal 10 FPS, including the TTSA3R part of E3 and the checks. The machine is shared and busy, so wall-clock may be longer. If the queue is projected to end after 2026-11-14, stop and report rather than cutting arms silently.

**Every outcome is reported, whichever way it goes.** Scenes are not excluded after the fact. Any deviation goes in a dated section below.

---

## Amendment 1 — 2026-11-06, before any full run

Committed after the short checks below and before any full-length arm. The registered text above is unchanged.

### What the checks showed

Run on one 90-frame TUM sequence (`walking_static`) and on KITTI sequence 04. These are the only ATEs of new arms in existence at this point, and all of them are in `results/relpose/tum__smoke/`.

| Check | Result |
|---|---|
| Baselines on this code | `ddd3r` 0.00689, TTT3R 0.01062, TTSA3R 0.00829, all identical to the logged values |
| Single-sequence ATE of new arms, not used for any verdict | `normmatch` 0.00793, `prevdir` 0.00702, `ttt3r_frozen` 0.01041 |
| **E1 manipulation check** | **Fails on this sequence.** Mean gain 0.211 for `ddd3r` against 0.367 for `normmatch` |
| **E5 check** | **Fails.** CUT3R 21.85 against the report's 32.45, brake τ=2 7.05 against 6.66 |
| Table 5 provenance | Brake τ=2 reproduces the logged 7.05001 exactly. τ=1 gives 5.55621. Table 5's KITTI 04 brake is therefore τ=2 |

### Why the E1 check fails

The rule is closed-loop. Under `ddd3r` the state barely moves along the drift, so the next delta stays aligned with it and the gain keeps shrinking. Under `normmatch` the state moves along the raw delta, so alignment and gain settle elsewhere. The arm therefore does not hold the norm at the directional rule's value, and it cannot answer R2-Q1 as asked.

### E1b — exact norm replay (new arm, primary for "same norm")

**Arm `normreplay`.** A `ddd3r` run on the same sequence logs each token's applied update norm per frame, `‖S_t − S_{t−1}‖`, which includes that run's gate. The replay moves each token by exactly that logged norm along its own raw delta: `S_t = S_{t−1} + ‖S^{ortho}_t − S^{ortho}_{t−1}‖ · δ̂_t`. Per token and per frame the applied norm equals the directional run by construction, so any difference in ATE comes from direction.

**Predictions.** Unchanged in direction, and stated again for this arm.

- **P1'.** On TUM, `ddd3r` is better than `normreplay`.
- **P2a'.** On ScanNet, `normreplay` is better than `ddd3r`.
- **P2b'.** On ARKitScenes, `normreplay` is better than `ddd3r`.

The outcome table of E1 applies with `normreplay` in place of `normmatch`, and **the verdict on direction is taken from `normreplay`**. `normmatch` still runs and its P1, P2a and P2b are still reported, as the performance of a closed-loop scalar rule. The first-five ScanNet manipulation run is dropped, because the full `ddd3r` rerun logs the same data.

**Caveat to report with the result.** The replay preserves pace, not trajectory. Its state differs from the directional run's, so "direction" here means the whole closed-loop effect of the decomposition.

### Other changes

- The reference `ddd3r` arm is rerun with logging on TUM, ScanNet and ARKitScenes, since the replay needs those logs. Verdicts for E1b compare against these fresh runs.
- ScanNet arms run only on the 65 scenes valid in the existing TTT3R log. The other 31 fail on non-finite ground truth under every method, so this exclusion does not depend on any result.
- Speed measured on this machine is about 5 FPS, not 10, so the estimate rises to about 25 GPU-hours. Three workers should finish in about 10 hours, inside the 2026-11-14 limit.
- E5's full run is stopped as registered and goes to the author for a decision.

---

## E5 check — result, 2026-11-06

**The registered check fails, so the full E5 run did not start.** KITTI sequence 04, ATE RMSE, same data in all three columns. Rebuilding `kitti_odo_full` here gives the same numbers as the logged `kitti_odom` runs, so the data preparation is not the cause.

| Arm | Report (origin/zjc, H200) | Compiled RoPE, this machine | Python fallback with clamp, this machine |
|---|---|---|---|
| CUT3R | 32.45 | 21.85 | 31.31 |
| brake τ=2 | 6.66 | 7.05 | 6.90 |
| ortho | 10.28 | 17.79 (logged) | 12.28 |

**Finding.** Commit `cd57a18` on `origin/zjc` ("fix: guard rope fallback indices") clamps positions in the Python RoPE fallback. The pose token's position is −1 (`src/dust3r/model.py`, `pose_pos_i = -torch.ones`), so the clamp maps it to 0. Without the clamp the fallback crashes, and the compiled extension handles −1 directly. All three arms are closer to the fallback column than to the compiled one.

**Status of this conclusion.** Strong evidence, not proof. The ortho gap to the report is still about 19%, and the report's own speed note (9–10 FPS on an H200) was not explained. The author of the report can settle whether their run had the compiled extension.

**Table 5 provenance, as registered.** Brake τ=2 reproduces the logged sequence 04 value of 7.05001 exactly and τ=1 gives 5.55621, so Table 5's KITTI 04 brake entry is τ=2 under compiled RoPE. Sequence 03 is queued.

**Fresh run of the instrument check.** The first TUM sequences of the full `ddd3r` rerun match the log to every printed digit.

---

## Amendment 2 — 2026-11-06, KITTI rerun approved by the author

Committed before any of the arms below started on the full sequences.

### E5b — all five methods on compiled RoPE

**Why.** The report's KITTI numbers are not comparable with the rest of the paper (E5 check above), so the whole column is rerun rather than only brake.

**Arms.** `cut3r`, `ttt3r_plain`, `const05` (α=0.5), `brake_t1` (τ=1), `ortho_plain` on `kitti_odo_full`, 11 sequences, full length, compiled RoPE. **No directional prediction**, as registered in E5.

**Protocol check after the run.** Sequences 03 and 04 must match the earlier compiled-RoPE values within 0.5%. A miss blocks every ruling below until explained.

| Arm | Seq 04 | Seq 03 |
|---|---|---|
| `cut3r` | 21.84807 | 150.29048 |
| `ttt3r_plain` | 11.06465 | 103.25776 |
| `const05` | 5.09581 | 60.68427 |
| `brake_t1` | 5.55621 | not run before |
| `ortho_plain` | 17.78537 | 39.90691 |

**Reported.** Per-sequence and average ATE RMSE and ATE mean, paired tests over the 11 sequences by the common rule, and official `t_rel` and `r_rel` recomputed from the new trajectories with `eval/relpose/eval_kitti_odom.py`.

**TTSA3R.** The earlier run in `rebuttal/results/relpose/kitti_odom/ttsa3r` fills the table's empty row only if sequence 04 reruns to the same value within 0.5%.

**Consequences, fixed now.**

- Table 2's KITTI column, the ranking table and Table 5's KITTI columns are replaced by these runs, whichever way the numbers go. Table 2 is labelled with the metric it actually prints.
- Recommendation (iii) and the sentence calling KITTI the strongest domain-shift test keep their claims only if the new numbers support them. If `ortho_plain` is not better than all four others by the common rule, "best aggregate for outdoor driving" is withdrawn.
- The 20.1%, 7.6% and 58.6% figures are replaced by the recomputed values or removed.
- `kitti_odo_full_report.md` is no longer a source for any number in the paper.

### Pose-only outputs

A shared-machine risk turned up: full per-frame outputs peak at about 44 GB for a 1000-frame sequence, so three workers would exceed the memory available. All later arms use `--pose_only`, which keeps only camera poses and leaves the state update untouched.

Checked: sequence 04 gives 21.84807 and 7.05001, identical to the full-output runs, and peak memory falls to 6.6 GB. The TUM `ortho_check` that was already running keeps full outputs.

### Table 5 provenance, completed

Brake τ=2 on KITTI sequence 03 gives 25.37579, matching Table 5's 25.38. Both KITTI columns of Table 5 are τ=2 under compiled RoPE.

### Time estimate

A 1000-frame sequence takes about 7 minutes here, so the "about 10 hours" in Amendment 1 is too low. The longest worker should take about 30 hours, ending around the morning of 2026-11-08 and inside the 2026-11-14 limit. E5b adds about 8 GPU-hours as a fourth worker.

---

## E5b — result, 2026-11-07

Everything below is produced by `scripts/kitti_e5b.py` from the logs, and printed in `results/kitti_e5b.txt` and `results/kitti_official_metrics.txt`.

### Protocol check

Seven of the nine registered references match, the largest gap being 0.00001%. Both `const05` references miss, by 39.3% on sequence 04 (5.09581 registered, 7.09814 rerun) and 21.8% on sequence 03 (60.68427 registered, 73.93217 rerun).

**The miss is in the registered references, not in the rerun.** They were copied from the logged `constant` folders on the assumption that those were α=0.5. Running α=0.33 on both sequences reproduces those folders in every printed digit of ATE and of both RPEs.

| Seq | ATE | RPE trans | RPE rot |
|---|---|---|---|
| 03 | 60.68427 | 2.30439 | 0.98484 |
| 04 | 5.09581 | 1.77483 | 0.65024 |

So the logged folders are α=0.33, and `const05` stands as registered at α=0.5. Amendment 2 blocks rulings until a miss is explained. It is now explained, so the rulings below proceed.

**New error this exposes.** Table 5's `\dddconst{}` row prints α=0.5 for TUM and ScanNet, and α=0.33 for both KITTI columns. Its 5.10 and 60.68 become 7.10 and 73.93. See S12 in `../ISSUE_BOARD.md`.

### Checks added after the run

These two were not registered in this form and carry no prediction.

- **TTSA3R.** Sequence 04 reruns to 7.75980 against the earlier 7.759798, which passes the registered 0.5%. The empty row is filled with the 11-sequence ATE RMSE of 174.30. That folder holds no trajectories, so TTSA3R has no segment metrics.
- **Table 5 γ rows.** Amendment 2 replaces Table 5's KITTI columns, but the five arms cover only three of its seven rows. γ = 0.5, 1, 2, 3 were rerun on sequences 03 and 04, and all eight values match the logs in every printed digit, so those rows stay as printed.

### Results, 11 sequences, compiled RoPE

| Arm | ATE RMSE | vs CUT3R | ATE mean | t_rel (%) | r_rel (deg/100m) | RPE trans | RPE rot |
|---|---|---|---|---|---|---|---|
| CUT3R | 203.94 | — | 182.97 | 71.29 | 22.255 | 2.877 | 3.367 |
| TTT3R | 180.23 | −11.6% | 157.96 | 68.30 | 20.603 | 3.516 | 4.165 |
| TTSA3R | 174.30 | −14.5% | 152.23 | n/a | n/a | not computed | not computed |
| const α=0.5 | 178.24 | −12.6% | 158.77 | 62.97 | 17.817 | 4.015 | 6.238 |
| brake τ=1 | 172.21 | −15.6% | 154.19 | 62.06 | 17.661 | 5.245 | 6.484 |
| ortho | 160.98 | −21.1% | 143.30 | 53.77 | 11.881 | 8.042 | 5.941 |

**Ortho against the other four, by the common rule.**

| Metric | vs CUT3R | vs TTT3R | vs const | vs brake |
|---|---|---|---|---|
| ATE RMSE | better, p=0.0020 | tied, p=0.1748 | tied, p=0.3203 | tied, p=0.2783 |
| t_rel | better, p=0.0010 | better, p=0.0068 | tied, p=0.0830 | tied, p=0.0830 |
| r_rel | better, p=0.0010 | better, p=0.0020 | better, p=0.0186 | better, p=0.0029 |

Also by the common rule on ATE RMSE: brake better than CUT3R (p=0.0186), TTSA3R better than CUT3R (p=0.0020), and TTT3R and const tied with CUT3R (p=0.0830, p=0.0537). Every pair among TTT3R, TTSA3R, const, brake and ortho is tied.

### Rulings, as fixed in Amendment 2

1. **"Best aggregate for outdoor driving" is withdrawn.** Ortho has the lowest mean but is better than only one of the four others on ATE.
2. **"Our strongest test of domain shift" is withdrawn.** Nothing here compares domain shift across datasets, and absolute error stays large for every method (t_rel above 50%).
3. **Figures replaced.** 20.1% becomes 21.1%, now on the RMSE the paper states. 7.6% becomes 24.6%. 58.6% becomes 46.6%.
4. **Table 2's KITTI column** becomes 203.94, 180.23, 174.30, 178.24, 172.21, 160.98, in ATE RMSE.
5. **Ranking table, KITTI column.** Const falls from 2nd to 3rd and brake rises from 4th to 2nd. The sentence "brake ranks 1st on ScanNet but 4th on KITTI" no longer holds, and KITTI no longer supports the ranking reversal, since the three operating points are tied there.
6. **Table 5.** Const 7.10, 73.93. Brake 5.56, 57.07. The other rows are unchanged. "With brake a close second" on sequence 03 no longer holds.
7. **`kitti_odo_full_report.md`** is no longer a source for any number in the paper.

All of this is drafted in `patches/tau_kitti_fix.patch`. It is not applied.

### Limits of these results

- **Multiple comparisons.** The common rule has no correction, and this run makes many comparisons. The r_rel result, the only one where ortho beats all four, survives Holm within those four tests (largest adjusted p=0.0186). It was not predicted in advance, as E5 registered no direction.
- **"Official" is not accurate.** `eval_kitti_odom.py` aligns with Sim(3) first and averages per-sequence means, whereas the KITTI devkit pools segments and does not rescale. The draft therefore no longer says "official".
- **The change from −7.6% to −24.6% is not all RoPE.** The report's tool is not in this code path, and its t_err for CUT3R is 93.94 against 71.29 here while r_err barely differs (22.66 against 22.26). The two tools may differ, so the sizes should not be compared.
- **One sequence dominates ATE.** Sequence 01 alone is 480–631 m. Ortho is second worst there (620.41 against brake's 479.88).
- **Local accuracy goes the other way.** TTT3R, const, brake and ortho all have worse frame-to-frame RPE than CUT3R, and ortho's translational RPE is 2.8 times CUT3R's. This is reported beside the segment metrics rather than left out.
- **The fallback conclusion is still unconfirmed by zjc,** but no number in the revised paper depends on it now.

---

## E1–E4 — result, 2026-11-07

All 27 queued arms exited 0 and every arm covers its full sequence list (8 TUM, 65 ScanNet, 18 ARKitScenes, 20 ScanNet for E3). Everything below is produced by `scripts/e1_e4_rulings.py` from the logs and printed in `results/e1_e4.txt`.

### Instrument check

Passes. The fresh TUM `ortho_check` gives 0.05454 against the registered 0.05454. On all three datasets the fresh `ortho_check` matches the logged `ddd3r` per sequence, with a largest gap of 0.0000%.

### E1b — exact norm replay, the verdict on direction

| Dataset | ortho | normreplay | gap | ortho wins | 95% CI (ortho − replay) | p | Rule | Prediction |
|---|---|---|---|---|---|---|---|---|
| TUM | 0.05454 | 0.07437 | −26.7% | 8/8 | [−0.0288, −0.0115] | 0.0078 | ortho better | P1' holds |
| ScanNet | 0.48829 | 0.31858 | +53.3% | 11/65 | [+0.1154, +0.2375] | 1.7e−10 | replay better | P2a' holds |
| ARKitScenes | 0.60927 | 0.54458 | +11.9% | 4/18 | [−0.0049, +0.1480] | 0.0526 | tied | **P2b' fails** |

**Ruling.** This is the registered row "P1 holds, P2a or P2b fails". Direction helps on TUM. The failure is reported, and **the claim that drift energy sets the sign of the directional effect is dropped**. ARKitScenes leans the same way as ScanNet and misses p<0.05 narrowly, but the rule is the rule.

### E1 — `normmatch`, the closed-loop scalar rule

| Dataset | ortho | normmatch | ortho wins | p | Rule | Prediction | Mean gain, ortho / normmatch | Gain gap |
|---|---|---|---|---|---|---|---|---|
| TUM | 0.05454 | 0.07608 | 7/8 | 0.0156 | ortho better | P1 holds | 0.3577 / 0.4533 | 26.7% |
| ScanNet | 0.48829 | 0.28575 | 9/65 | <0.001 | normmatch better | P2a holds | 0.3558 / 0.4449 | 25.0% |
| ARKitScenes | 0.60927 | 0.52753 | 4/18 | 0.0198 | normmatch better | P2b holds | 0.3743 / 0.4596 | 22.8% |

**The manipulation check fails on all three datasets** (gap above 10%), as Amendment 1 expected, so this arm does not answer "same norm". It is reported only as the performance of a closed-loop scalar rule, and the verdict on direction stays with `normreplay`.

### E2 — projection timing, no prediction

| Dataset | updated d_t (ortho) | d_{t−1} (prevdir) | updated wins | p | Rule |
|---|---|---|---|---|---|
| TUM | 0.05454 | 0.06156 | 8/8 | 0.0078 | updated better |
| ScanNet | 0.48829 | 0.47645 | 37/65 | 0.8410 | tied |

### E3 — token-level temporal adaptation

**(a) Measurement**, averaged over sequences.

| Data | Method | Token | Frame | Residual | Per-token temporal std | Frame-average std |
|---|---|---|---|---|---|---|
| TUM | TTT3R | 37.1% | 13.2% | 49.7% | 0.0375 | 0.0168 |
| TUM | TTSA3R | 8.8% | 4.7% | 86.6% | 0.0852 | 0.0205 |
| ScanNet first 20 | TTT3R | 23.6% | 27.1% | 49.3% | 0.0507 | 0.0305 |
| ScanNet first 20 | TTSA3R | 9.0% | 2.8% | 88.2% | 0.1018 | 0.0191 |

TTSA3R's logged gate is the fused TAUM × SCUM mask. The paper's σ≈0.006 concerns TAUM alone, so the two are not the same quantity.

**(b) Functional test.** A = frozen, B = live.

| Data | Method | Frozen | Live | Frozen better on | 95% CI | p | 5% margin | Verdict |
|---|---|---|---|---|---|---|---|---|
| TUM | TTT3R | 0.11905 | 0.10251 | 5/8 | [−0.0040, +0.0393] | 0.7422 | 0.0051 | **inconclusive** |
| TUM | TTSA3R | 0.08840 | 0.09075 | 6/8 | [−0.0058, +0.0008] | 0.1484 | 0.0045 | **supported** |
| ScanNet first 20 | TTT3R | 0.38923 | 0.41069 | 15/20 | [−0.0355, −0.0067] | 0.0064 | 0.0205 | **supported** |
| ScanNet first 20 | TTSA3R | 0.38028 | 0.38220 | 8/20 | [−0.0432, +0.0280] | 0.4553 | 0.0191 | **inconclusive** |

**Ruling.** Nothing is refuted, so nothing has to be withdrawn by the registered rule for a single method and dataset. Two cells are inconclusive, and for those the paper may state the measurement only. "No meaningful temporal adaptivity" and "rules out a useful timing role" cover all four cells, so they cannot stand as written. They are narrowed to the frame-averaged statistic plus this test.

The precedence choice in `frozen_verdict` never came into play, since no cell met both criteria. TTT3R's +16.1% on TUM comes from three sequences (sitting_halfsphere +0.040, walking_halfsphere +0.054, walking_xyz +0.072), and the other five move slightly the other way.

### E4 — brake against a constant of the same scale

ScanNet `const033` 0.26237 against brake τ=1 0.26147: const wins 33/65, CI [−0.0066, +0.0081], p=0.9505, tied. **P4 holds.** With TUM (0.0659 against 0.0634, p=0.55) brake and a constant multiplier of 0.33 are tied on both datasets, and brake's multiplier at the measured cosines is sigmoid(−0.62…−0.77) = 0.35…0.32. So **brake's gain over const 0.5 is a scale effect and must not be credited to temporal adaptivity.** A tie is not equivalence. The ScanNet interval is narrow, about ±3% of the mean, and TUM's is not.

### Comparisons outside the registration, no prediction

- `normreplay` against TTT3R: TUM 8/8, p=0.0078, better. ScanNet 46/65, p=0.0015, better. ARKitScenes 10/18, p=0.59, tied.
- `normreplay` against brake: TUM and ScanNet worse (p=0.0156, p<0.001), ARKitScenes tied.
- Ortho on ScanNet: worse than TTT3R (29/65, p=0.047) and TTSA3R (27/65, p=0.0035). On ARKitScenes: tied with TTT3R (p=0.14) and brake (p=0.078), worse than const 0.5 (p=0.028).
- Of the 59 logged configurations with results on both TUM and ScanNet, none has a lower mean than brake on both. Means only, and some of those runs predate the final code.

### What this does to the paper

1. Keep the directional claim, scoped: it helps on TUM. It is not a general improvement.
2. Drop "drift energy sets the sign". This touches `sec/analyse.tex:127` ("consistent signed signal") and the abstract's "where drift carries mostly harmful repetition".
3. Narrow the gate-collapse wording in the abstract and at `sec/analyse.tex:69`.
4. Do not credit brake's gain over const 0.5 to temporal adaptivity.

### Limits of these results

- **Multiple comparisons.** The common rule has no correction, and this report makes many comparisons.
- **The replay preserves pace, not trajectory.** Its state differs from the directional run's, so "direction" means the whole closed-loop effect of the decomposition, as registered.
- **ATE is not the whole picture.** Ortho's mean RPE rot on TUM is 0.652 against 0.450 for the replay, so its ATE gain comes with worse frame-to-frame rotation.
- **Frozen gates are an analysis device.** They use each sequence's own future, so they are not a deployable method.
- **E3 on ScanNet covers 20 of 65 scenes**, as registered.

---

## Correction, 2026-11-07

Checked against the submitted PDF. The relative-pose table whose KITTI column this file calls "Table 2" is **Table 3** in that PDF, since Table 2 there is the directional statistics. Table 5 and the appendix numbers are as written. The registered text above is left as it was.

The line in the E1–E4 section saying that none of "59 logged configurations" is lower than brake on both datasets is uninformative and is not used in the reply. These are runs rather than distinct configurations, since at least `ttt3r_momentum_inv_t1` is brake itself. No logged run is lower than brake on ScanNet alone, while 33 are lower on TUM, so brake is a selected winner. Both counts are now printed in `results/e1_e4.txt`, along with brake against TTT3R, which ties on ARKitScenes (10/18, CI [−0.0589, +0.0245], p=0.286).
