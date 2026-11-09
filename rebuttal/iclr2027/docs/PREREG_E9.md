# E9: a drift-energy routing rule, tested on held-out datasets

> **Fixed before any E9 value existed.** Its SHA-256 is written to `results/e9_registration.sha256`, and `scripts/run_e9.sh` refuses to start if the file no longer matches. Nothing above the first line holding only `---` may be edited after results exist. Corrections and results go in dated sections after it.

Submission: ICLR 2027, 28147. Registered 2026-11-09. Follows `PREREG_E8.md`. The comparison rule, the arm definitions in `scripts/run_arm.sh` and the code checked by `results/e7_code.sha256` apply unchanged.

## Why this exists

qpVA W1 asks how to choose a configuration for unseen scenes. The replies and the revision say we have no validated rule. An exploratory analysis on 2026-11-09 (not registered, scripts in the session scratchpad) found one candidate. Per-sequence selection by whole-sequence drift energy does not work, since within-dataset correlations between drift energy and ortho's advantage have mixed signs (TUM +0.34, ScanNet −0.38, ARKitScenes +0.53). A coarser rule does separate TUM from ScanNet: route a sequence to ortho when its drift energy over the first 200 frames is below 0.5, and to brake otherwise. In sample this matched the better fixed choice on TUM and ScanNet. E9 asks whether the rule, unchanged, holds on datasets that played no part in choosing it.

## The rule, fixed

**Statistic ē(s).** For a sequence s, take the entries of its evaluated input directory (below) sorted by name, as `eval/relpose/launch.py` sorts them, keep those whose lower-case name ends in `.png`, `.jpg` or `.jpeg`, as both image loaders do, and keep the first n = min(200, T) of these T files. Run `analysis/a4_drift_scaling.py::analyze_scene` on them exactly: the checkpoint `model/cut3r_512_dpt_4_64.pth` with `model_update_type = "ttt3r"`, images loaded by `dust3r.utils.image.load_images(paths, size=512)`, one call to `forward_recurrent_analysis`. Its `cosine_history[t]` is the cosine between δ_t and δ_{t−1}, averaged over state tokens, where δ_t is the candidate state minus the current state. Then ē(s) = mean over t = 2..n of cosine_history[t]². This is the quantity in the paper's Table 2 (TUM 0.398, ScanNet 0.598).

**Routing.** ē(s) < 0.5 → ortho (α⊥ = 0.5, α∥ = 0.05, β_ema = 0.95, γ = 0). ē(s) ≥ 0.5 → brake τ = 1. The rule's result on s is the chosen arm's full-length result on s. Deployed, this means one extra TTT3R pass over at most 200 frames before the chosen run starts.

**Evaluated input directories.** KITTI Odometry: `rebuttal/iclr2027/data/long_kitti_odo_s1/<seq>/image_full`. Sintel: `data/sintel/training/final/<seq>`, the 14 sequences of `eval/relpose/metadata.py`, all shorter than 200 frames, so n = T. ARKitScenes 500f: `data/long_arkit_s1/<scene>/rgb_500`.

## What was already known at registration

This registration is not blind to outcomes. Read before writing it:

- **Used to choose the rule.** Prefix-200 ē for the 8 TUM sequences (0.313 to 0.454, all below 0.5) and the 96 ScanNet scenes (0.421 to 0.698, five below 0.5, three of them among the 65 valid at 1000 frames). Whole-sequence ē for ARKitScenes at 500 frames (mean 0.632, s.d. 0.035, none below 0.5) and TUM and ScanNet at 1000 frames. All E-series outcome arms on TUM 1000f, ScanNet 1000f and ARKitScenes 500f. Thresholds 0.45 and 0.50 were both tried on TUM, ScanNet and ARKitScenes, with brake as the high branch.
- **KITTI Odometry, 11 sequences.** Every outcome arm of E5b, E6 and E7, per sequence: ATE, r_rel, t_rel and the 1-frame RPEs. Between the two branches, ATE ties (ortho 160.98, brake 172.21, p = 0.28) and r_rel favours ortho (11.88 vs 17.66 deg/100m, p = 0.0029, computed while writing this). **No drift energy of any kind has been computed on KITTI Odometry.**
- **Sintel.** The paper's Table 4 Sintel column, from an older code version: TTT3R 0.209, brake 0.237, ortho 0.236, const 0.220. Per-sequence values were not read. No drift energy has been computed on Sintel.
- **ARKitScenes.** Ortho ties brake in ATE (0.609 vs 0.539, p = 0.078). Its whole-sequence ē was used to choose the threshold, so ARKitScenes is **partly seen**. It is reported beside the test and never counted.

So the held-out test sets are **KITTI Odometry and Sintel**. On KITTI the outcomes of both branches are known and only the routing is not, so E9 there tests whether the prefix ē of KITTI sequences falls on the side of 0.5 that picks ortho, where ortho has the lower segment rotation error.

## Protocol

**Code.** Unchanged from E7. `scripts/run_e9.sh` checks `/usr/bin/git diff 43f7e29 -- src eval` against `results/e7_code.sha256` and that the compiled RoPE is present, and refuses to start otherwise. The statistic is computed by `scripts/e9_selector.py prefix` and the ruling by `scripts/e9_selector.py rule`.

**New runs.** Sintel, 14 sequences, dataset key `sintel`, output group `sintel`: `ortho_plain` and `brake_t1` (the two branches), `ttt3r_plain` for context and for check I2. KITTI Odometry and ARKitScenes use existing arms: KITTI `kitti_odo_full/ortho_plain` and `kitti_odo_full/brake_t1`, with r_rel from `results/kitti_official_metrics.txt`. ARKitScenes uses `arkit_s1_500/ortho_check` and the logged brake τ = 1 run used in all replies.

**Cells.** A cell is a dataset and a metric. Held-out cells: KITTI ATE, KITTI r_rel, Sintel ATE. Partly seen: ARKitScenes ATE.

**Comparison rule.** From `scripts/stats_rule.py`, as in E7 and E8. "A is better than B" needs a lower mean, p < 0.05 and a gap above 2.2%. Anything else is "tied". A cell is **informative** when one branch is better than the other there. At registration, KITTI r_rel is informative and KITTI ATE is not. Sintel ATE is decided from the new runs.

### Checks

- **I1, the statistic.** Before any held-out ē is computed, `e9_selector.py prefix` recomputes ē for the 8 TUM sequences (first 200 frames of `rgb_1000`, as the stored values used) and the five ScanNet scenes whose stored ē is below 0.5 (first 200 frames of `color_200`). Each must be within 0.005 of `analysis_results/a4_drift_scaling/{tum,scannet}_200f.json` and on the same side of 0.5. A miss stops E9 before any held-out ē exists.
- **I2, the Sintel arms.** Each Sintel arm yields a finite ATE on all 14 sequences, and `ortho_plain` and `brake_t1` each differ from `ttt3r_plain` on at least one sequence, so the update type reached the model. The mean ATE of `ttt3r_plain` is compared with the earlier run in `eval_results/relpose/sintel/ttt3r` (read only after this registration) and the gap is reported. It does not stop E9. A failure of the first two conditions stops E9 until it is explained in a dated correction.
- **Sequence counts.** A sequence whose prediction and ground truth differ in length is reported and left out of that cell. Nothing else is excluded.

## Predictions

- **P9a.** In every informative held-out cell, the rule is not worse than the better branch.
- **P9b.** In both held-out ATE cells, the rule is not worse than brake.

**E9 holds** when I1 and I2 pass, at least one held-out cell is informative, and P9a and P9b both hold. **E9 fails** otherwise. With KITTI r_rel informative at registration, E9 fails if most KITTI sequences route to brake.

Reported with no prediction: per-sequence ē and routing, the number of sequences with ē within 0.005 of 0.5, every pairwise comparison among the rule, ortho, brake and TTT3R in every cell, and the same for ARKitScenes.

## Consequences for the replies and the revision

| Outcome | Consequence |
|---|---|
| I1 or I2 fails | No ruling. Fix the cause, record it in a dated correction, rerun what failed |
| E9 holds | May say, in a follow-up comment and in the appendix, that a rule fixed on TUM and ScanNet (ortho when the drift energy over the first 200 frames under TTT3R is below 0.5, brake otherwise) was applied unchanged to KITTI Odometry and Sintel, and was not worse than the better of the two branches in any metric where they differ there, nor worse than brake in ATE. The same text must say: two held-out datasets of 11 and 14 sequences, a threshold chosen after looking at TUM, ScanNet and ARKitScenes, one extra TTT3R pass over 200 frames, and a heuristic, not a validated selector. "We have no validated rule" may become "our one rule held on two held-out datasets, too few to call it validated". The default recommendation does not change |
| E9 fails | "No validated rule" stays everywhere. E9 goes to `REVISION_LOG.md` and stays out of the paper. No other threshold, statistic or prefix length is tried on KITTI Odometry, Sintel or ARKitScenes for this rebuttal |
| ARKitScenes, any result | Reported beside the test, never counted and never called held-out |

Under every outcome no text may say that drift energy predicts the sign of the direction effect, and nothing here changes the default configuration or makes the rule the paper's method.

## Budget and stopping

The statistic needs 2600 frames for I1 and 6443 for the held-out and partly seen sets (KITTI 2200, Sintel 643, ARKitScenes 3600), and each Sintel arm 643 frames. About 1 GPU-hour on a shared GPU in total. If E9 has not finished by 2026-11-10 12:00 CST, stop, rule on what finished, and report the rest as not run.

---

## Results, 2026-11-09

Run 10:07 to 10:46 CST, GPU 1 for the statistic and GPU 0 for the Sintel arms. `scripts/e9_selector.py` had SHA-256 e8a59ae5… (`results/queue_e9.log`). Numbers in `results/e9_instrument.txt`, `results/e9_prefix.json` and `results/e9.txt`. Nothing deviated from the registration.

**Checks.** I1 passes: all 13 recomputed values equal the stored ones to five decimals. I2 passes: every Sintel arm has 14 finite ATEs, and both branches differ from TTT3R on 14 of 14 sequences. TTT3R's Sintel mean is 0.20783 against 0.20879 for the earlier run (−0.46%). Every Sintel sequence has one pose per image in every arm, so nothing was excluded.

**Routing.** KITTI: mean ē 0.501, range 0.463 to 0.546, 5 of 11 to ortho (02, 04, 06, 07, 08), 2 within 0.005 of 0.5 (07 at 0.497, 09 at 0.500). Sintel: mean 0.497, range 0.371 to 0.700, 10 of 14 to ortho, none near 0.5. ARKitScenes: mean 0.620, range 0.535 to 0.687, 0 of 18 to ortho.

| Cell | Ortho vs brake | Rule vs better branch | Rule vs brake |
|---|---|---|---|
| KITTI r_rel, informative | ortho better, 11.88 vs 17.66, p=0.0029 | **worse than ortho**, 15.24 vs 11.88 (+28.3%), wins 0/11, p=0.031 | tied, p=0.19 |
| KITTI ATE | tied, p=0.28 | — | tied, 162.80 vs 172.21, p=0.31 |
| Sintel ATE | tied, 0.2361 vs 0.2372, p=0.30 | — | tied, 0.2352 vs 0.2372, p=0.49 |
| ARKitScenes ATE, not counted | tied, p=0.078 | — | identical, every scene routed to brake |

**Ruling.** One held-out cell is informative. P9a fails: on KITTI r_rel the rule is worse than ortho, because the 6 sequences routed to brake each have a higher r_rel under brake (by 2.8 to 14.1 deg/100m). The 28.3% gap is larger than the largest rounding-noise shift measured on KITTI r_rel (13.4%). P9b holds. **E9 fails.**

Reported with no prediction: the rule beats TTT3R on KITTI r_rel (15.24 vs 20.60, p=0.019), and on Sintel no regulated arm beats TTT3R (rule 0.2352 vs 0.2078, tied, p=0.17).

**Consequence, as registered.** "No validated rule" stays in the replies and the revision. E9 is recorded in `REVISION_LOG.md` and stays out of the paper. No other threshold, statistic or prefix length is tried on KITTI Odometry, Sintel or ARKitScenes for this rebuttal.
