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
