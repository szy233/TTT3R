# STRATEGY PLAN — Submission 8591 (DDD3R)

## Situation

Three reviewers, **all at Rating 4 (Borderline accept)**. Nobody is hostile; the paper's diagnosis
and framing are praised across the board (Clarity 4/4/4). The gap to acceptance is entirely
**evaluation scope**, not correctness or novelty. Two reviewers made explicit conditional promises
to raise their scores. This is a winnable rebuttal.

**Critical asymmetry**: the single most damaging weakness (DpBu's "complete absence of dynamic-scene
evaluation", which he rates as blocking) rests on a **factually incorrect premise**. We already
evaluate on three dynamic benchmarks — we just never said so. Correcting this costs zero experiments
and converts a critical weakness into a strength.

---

## Global Themes (opener)

**T1 — "Dynamic evaluation is already there; we under-sold it."**
TUM (all 8 seqs = freiburg3 sitting/walking, TUM RGB-D's *Dynamic Objects* category), Bonn
(= Bonn RGB-D *Dynamic*: balloon/crowd/person_tracking), and Sintel (large character motion) are
all dynamic-scene benchmarks. 3 of 6. Plus a **new dynamics-severity split** computed from existing
raw output showing DDD3R holds under severe dynamics. Answers R2-C2 outright and reinforces R1-C1.

**T2 — "Scalar gating cannot fix direction — and that is provable in one line."**
For any scalar gate g_t, the applied update g_t·δ_t is parallel to δ_t, so the drift-energy fraction
cos²(δ_t, d_t) is **invariant to g_t**. No amount of gate adaptivity changes directional composition.
This is the theoretical statement R1 asks for (Q2), it subsumes the empirical M2 collapse findings,
and it is the formal justification for the two-coefficient design. Pairs with existing Appendix A.8.

**T3 — "The method has 3 hyperparameters; the recommended default has 1."**
The ~15 `auto_gamma` variants are ablation instruments we report for completeness, not knobs the
user must set. DDD3R_brake (τ only) ranks 1st/2nd everywhere. Sensitivity is flat (<2% over
α∥ ∈ [0.05, 0.20]). Answers R1-C3, and reframes R2-C3.

**T4 — "Scope of the backbone claim, stated honestly."**
DUSt3R/MASt3R have no recurrent state, so DDD3R is inapplicable to them *by construction*, not
merely untested. We scope the claim to recurrent-state models and say what would falsify it.

---

## Response Mode per Issue

| ID | Mode | Cost | Owner |
|---|---|---|---|
| R2-C2 dynamic | direct_clarification + grounded_evidence (**new table, already computed**) | 0 | done |
| R1-C3 hyperparams | direct_clarification + narrow_concession | 0 | done |
| R2-C3 online γ | narrow_concession + grounded_evidence (negative-result sweep) | 0 | done |
| R1-C2 theory | assumption_hierarchy + user_confirmed_derivation | low (writing) | **needs sign-off** |
| R3-C1 related work | nearest_work_delta (citations verified ✓) | low (writing) | ready |
| R1-C1 real-world | grounded_evidence (KITTI OOD) + optional demo | 0 / low | **decision** |
| R3-C2 TAUM row | grounded_evidence (gate stats now; ATE run if approved) | **low-med** | **decision** |
| R2-C1 / R3-C3 backbone | direct_clarification + future_work_boundary (+ run if feasible) | high | **decision** |

---

## Citation Verification (R3-C1) — all confirmed, reviewer made an error

| Reviewer ref | Verified record | Status |
|---|---|---|
| [1] ZPressor | Wang et al., *ZPressor: Bottleneck-Aware Compression for Scalable Feed-Forward 3DGS*, **NeurIPS 2025**, arXiv:2505.23734 | ✓ exact |
| "iLRM" (prose) | Kang, Nam et al., *iLRM: An Iterative Large 3D Reconstruction Model*, arXiv:2507.23277 | ✓ **distinct paper** |
| [2] Long-LRM | Ziwen Chen, Hao Tan, Kai Zhang, Sai Bi, Fujun Luan, Yicong Hong, Li Fuxin, Zexiang Xu, *Long-LRM: Long-sequence Large Reconstruction Model for Wide-coverage Gaussian Splats*, **ICCV 2025**, arXiv:2410.12781 | ✓ exact |
| [3] survey | Wang, Weijie et al., *Feed-Forward 3D Scene Modeling: A Problem-Driven Perspective*, arXiv:2604.14025 (2026) | ✓ exact |

**Note**: the reviewer's prose names *iLRM* but ref [2] is *Long-LRM* — two different papers by
different author groups. We will cite **all three** plus the survey. Doing so demonstrates we
engaged with the suggestion carefully rather than pasting their list.

**Delta to articulate**: ZPressor / Long-LRM / iLRM attack redundancy along the **spatial-token /
capacity** axis — how many tokens carry the state, compressing *width*. DDD3R attacks redundancy
along the **temporal-directional** axis — which direction each update moves the state, regulating
the *trajectory*. Orthogonal and composable: a compressed state still accumulates drift, and
DDD3R would regulate it identically. Positioning this way strengthens rather than dilutes novelty.

---

## Character Budget (to be finalized once limit is known)

- Opener + global themes: 10–15%
- R3 (1ake): **~30%** — lowest Quality score (2), explicit promise to raise, cheapest asks
- R2 (DpBu): ~28% — dominated by the dynamic-scene correction (high impact, compact to state)
- R1 (ULz9): ~25% — confidence 5, so answers must be precise and non-defensive
- Closing (meta-reviewer): 5–10%

---

## Blocked Claims — nothing may be written until resolved

1. **Any claim of a cross-backbone experiment** — no second recurrent backbone exists in the repo.
   Blocked unless the user approves and completes a run.
2. **Any TAUM end-to-end ATE number** — only gate *statistics* (σ_time = 0.006) exist today.
   May state the collapse measurement; **may not** state a TAUM ATE unless the run is done.
3. **Any in-the-wild quantitative result** — demo assets exist but no metrics. Qualitative only.
4. **The scalar-gate invariance proposition** — trivially provable and already implicit in the
   paper's argument, but must be user-confirmed before being presented as a formal result.

---

## Risks

- **R1 has confidence 5.** Do not over-claim; they will check. Every number must trace to raw output.
- **The dynamic-scene correction must be phrased graciously.** The reviewer's premise is wrong, but
  it is wrong *because our appendix never labelled the datasets*. Frame as "we should have made this
  explicit", not "you didn't read". This is a genuine presentation failure on our part.
- **Do not oversell the γ negative results.** They are honest and useful, but three reviewers
  independently flagged operating-point selection — conceding it as an open problem is the right move.
- Reviewer 1ake's Quality 2 is out of step with their own text (which is largely positive) and with
  the other two reviewers' Quality 3. The related-work + TAUM fixes are cheap; prioritize them.
