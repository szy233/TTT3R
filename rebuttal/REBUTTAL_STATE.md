# REBUTTAL STATE — Submission 8591 (DDD3R)

- **Venue**: NeurIPS 2026 · **Format**: text-only · **Limit**: 6000 chars **per reviewer**
- **Stage**: initial rebuttal
- **Phase reached**: 4 (draft written) — Phases 5–7 (lints, stress test, finalize) pending experiments

## Reviewers

| ID | Rating | Quality | Conf | Note |
|---|---|---|---|---|
| ULz9 | 4 | 3 | **5** | wants theory + real-world; checks details |
| DpBu | 4 | 3 | 4 | "Quality could go to 4" if cross-backbone added |
| 1ake | 4 | **2** | 4 | "I will raise my score if you resolve my concern" |

## Draft budget (of 6000 each)

| Reviewer | chars | headroom |
|---|---|---|
| 1ake | 2942 | 3058 |
| DpBu | 3029 | 2971 |
| ULz9 | 2645 | 3355 |

Ample room remains for the pending results.

## Evidence ledger

| Claim | Source | Status |
|---|---|---|
| TUM = all dynamic (`freiburg3_sitting_*`/`walking_*`) | `data/long_tum_s1/` listing | ✅ verified |
| Bonn = Bonn RGB-D *Dynamic* (`balloon2/crowd2/crowd3/person_tracking2/synchronous`) | `eval_results/video_depth/bonn_s1_500/*/` | ✅ verified |
| TUM dynamics-severity split table | `rebuttal/scripts/make_tables.py` over raw `_eval_metric.txt` | ✅ reproduces paper's 0.055/0.166 exactly |
| Bonn 500f abs_rel per method | `result_scale&shift.json` | ✅ verified |
| TTSA3R = faithful reimplementation | line-diff vs `/home/szy/research/TTSA3R/src/dust3r/model.py:1142-1159` | ✅ verified |
| TTSA3R Sintel 0.2091 (n=14) | `rebuttal/results/relpose/sintel/ttsa3r/` | ✅ complete |
| TTSA3R TUM 1000f | run in progress | ⏳ 1/8 seqs |
| TTSA3R ScanNet 1000f | chained after TUM | ⏳ queued |
| Scalar-gate invariance proposition | derivation, user-approved | ✅ approved for use |
| ZPressor / Long-LRM / iLRM / survey citations | web-verified, see STRATEGY_PLAN | ✅ verified |
| 15+ failed online-γ signals | `docs/experiment_results.md` | ✅ in paper's own records |
| In-the-wild qualitative | not run | ⬜ E2 |
| Second recurrent backbone | not run | ⬜ E3 |

## Blocked claims (must not appear in a submitted rebuttal)

1. Any second-backbone result — **E3 not run**.
2. TTSA3R TUM/ScanNet numbers — **still running**; `[[PENDING]]` markers guard these.
3. Any in-the-wild quantitative claim — E2 would be **qualitative only** (no GT for the demo clips).

## Resource constraint

Only **GPU 1** is available (GPU 0 is another user's job — `zwf`, `run_classifier_lstm.py`).
GPU 1 is committed to E1: TUM 1000f (~8 seqs × 1000 frames), then ScanNet 1000f
(96 scenes × 1000 frames, several hours). E2/E3 must queue behind or interleave.

## Next actions

1. Wait for E1 TUM → fill `[[PENDING:E1_TUM]]`, check the registered prediction.
2. Wait for E1 ScanNet → fill `[[PENDING:E1_SCANNET]]`.
3. Decide E2/E3 scope given single-GPU contention.
4. Run Phase 5 lints (coverage/provenance/commitment/tone/consistency/limit).
5. Phase 6 stress test, then Phase 7 finalize `PASTE_READY.txt` + rich version.
