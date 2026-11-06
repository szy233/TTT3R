# REBUTTAL STATE — NeurIPS 2026 Submission 8591 (DDD3R)

- **Venue**: NeurIPS 2026 · **Format**: text-only · **Limit**: 6000 chars **per reviewer**
- **Stage**: initial rebuttal — **Phase 7 complete, ready to post**

## Deliverables

| File | Purpose |
|---|---|
| `PASTE_READY.txt` | three blocks, paste one per reviewer; macros expanded, counts verified |
| `REBUTTAL_DRAFT_v1.md` | working draft with LaTeX macros, fuller wording |
| `ISSUE_BOARD.md` | 9 atomized concerns, all closed |
| `STRATEGY_PLAN.md` | themes, budget, citation verification |
| `docs/E1_ttsa3r_verification.md` | TTSA3R reimplementation provenance |
| `docs/E3_streamvggt_findings.md` | cross-backbone architectural study + memory measurement |
| `docs/E4_arkitscenes_PREREGISTRATION.md` | pre-registration, amendment, locked prediction, results |

## Character counts (paste-ready, per reviewer)

| Reviewer | chars | headroom |
|---|---|---|
| 1ake | 4505 | 1495 |
| DpBu | 3578 | 2422 |
| ULz9 | 5481 | 519 |

## Reviewers

| ID | Rating | Quality | Conf | Their conditional |
|---|---|---|---|---|
| ULz9 | 4 | 3 | **5** | — |
| DpBu | 4 | 3 | 4 | "Quality could be increased to 4" if cross-backbone added |
| 1ake | 4 | **2** | 4 | "I will raise my score if you resolve my concern" |

## Coverage — 9/9 closed

| ID | Concern | Resolution |
|---|---|---|
| R1-C1 | real-world generalization | **E4 ARKitScenes**, pre-registered predictive test |
| R1-C2 | theory for TTT3R's limit | scalar-gate invariance proposition |
| R1-C3 | too many hyperparameters | 3 params; recommended configs have 1; flat sensitivity |
| R2-C1 | cross-backbone (MASt3R/DUSt3R) | not recurrent; **E3 StreamVGGT** + 3-family taxonomy |
| R2-C2 | dynamic scenes absent | premise corrected: 3/6 benchmarks dynamic + severity split |
| R2-C3 | online γ selection | conceded; 15+ negative results reported |
| R3-C1 | ZPressor / iLRM / Long-LRM | citations verified; axis-of-redundancy delta |
| R3-C2 | TTSA3R head-to-head | **E1**: full row, both main tables |
| R3-C3 | single backbone | → R2-C1 |

## Experiments run for this rebuttal

| ID | What | Status |
|---|---|---|
| E1 | TTSA3R as live update type; 8 eval cells | ✅ complete |
| E3 | StreamVGGT architecture + KV-cache growth (270 MB/frame measured) | ✅ complete |
| E4 | ARKitScenes, 18 scenes × 500f, pre-registered | ✅ complete |
| E2 | in-the-wild qualitative demo | ⏸ stopped — optional, no claim depends on it; machine contended |

## Pre-registration audit trail (git)

| commit | content |
|---|---|
| `2650914` | P1/P2/P3 registered **before** any ARKitScenes ATE |
| `895b610` | Amendment 1: 1000f → 500f, data-driven, still pre-ATE |
| `b74c4c0` | drift energy 0.632 measured; **P2 branch locked**, still pre-ATE |
| `fd48bf0` | results: P1 holds, P2 core holds / strict order fails, P3 fails |
| `f87fb61` | folded into draft; two factual errors fixed |

## Verification performed

- **Coverage lint**: 9/9 concerns anchored in the draft.
- **Provenance lint**: every number traced to raw eval output or a committed JSON.
- **Commitment lint**: caught and fixed a past-tense claim ("we have added a paragraph") for a
  paper edit that had not been made.
- **Consistency lint**: caught and fixed a cross-reviewer contradiction — the ULz9 corollary
  called scalar gates "empirically interchangeable" while the 1ake reply showed TTSA3R beating
  TTT3R everywhere.
- **Numeric audit**: ARKit means, drift energy, TUM dynamics split, StreamVGGT 270 MB/frame, and
  the 8/8 TUM claim all re-derived from raw files.
- **Two factual errors fixed**: (i) the false claim that brake ranks 1st/2nd on every benchmark —
  it is 4th on KITTI relpose; (ii) an inconsistent denominator in the ortho-vs-const gap.

## Known risks / open items

1. **Shared-machine incident.** An unscoped `pkill -f "demo.py"` (~20:27) may have killed an
   in-flight run of a colleague's `run_kitti_alpha_top.sh` on the same account. Cannot be
   confirmed either way. Their loop skips a killed sequence, so recovery is re-running that one
   sequence. **Recommend verifying seq 00/01 output with its owner.** All later kills were scoped
   to `-u szy -f <own path>`.
2. **Paper not yet edited.** Every "we will …" in the rebuttal is a camera-ready commitment, not a
   completed change. The related-work paragraph, dataset-characteristics table, dynamics split
   table, formal invariance subsection, negative-results sweep, and Sec. 4 restructure all remain
   to be written.
3. **KITTI relpose baselines are single-sequence.** `eval_results/relpose/kitti_odom/{cut3r,ttt3r}`
   contain only seq 04, while TTSA3R ran all 11. The full 11-sequence baselines live in the `zjc`
   branch report. The rebuttal therefore makes no KITTI TTSA3R claim.
4. **E2 not run.** If an in-the-wild qualitative figure is wanted for camera-ready, re-run
   `rebuttal/scripts/run_inwild_demo.sh` when the machine is free.
