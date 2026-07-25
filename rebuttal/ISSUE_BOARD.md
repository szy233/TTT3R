# ISSUE BOARD — NeurIPS 2026 Submission 8591 (DDD3R)

Reviewers: **ULz9** (R1, conf 5), **DpBu** (R2, conf 4), **1ake** (R3, conf 4).
All three: **Rating 4 (Borderline accept)**. Scores below.

| Reviewer | Quality | Clarity | Signif. | Orig. | Conf. | Stance |
|---|---|---|---|---|---|---|
| ULz9 | 3 | 4 | 3 | 3 | **5** | swing |
| DpBu | 3 | 4 | 3 | **4** | 4 | positive |
| 1ake | **2** | 4 | 3 | 3 | 4 | swing — *"I will raise my score if you resolve my concern"* |

**Score-movement priority: 1ake > ULz9 > DpBu.** 1ake gave the lowest Quality (2) and made an
explicit conditional promise to raise. DpBu also offered "Quality could be increased to 4".

---

## R1 — Reviewer ULz9

### R1-C1 — Limited dataset analysis / real-world generalization
- **raw_anchor**: "conclusions may not generalize well to real-world scenarios"; Q1 "How does DDD3R perform on in-the-wild datasets"
- **issue_type**: empirical_support · **severity**: major · **stance**: swing
- **response_mode**: grounded_evidence + direct_clarification
- **evidence available (no new runs)**:
  - 6 datasets / 3 tasks / 4 metrics already in paper: TUM, ScanNet, Sintel, KITTI Odometry, Bonn, 7Scenes
  - **KITTI Odometry is out-of-domain**: outdoor driving, 271–4661 frames, 11 sequences — the base checkpoint is trained on indoor-heavy short pairs. DDD3R still −20% ATE / −58.6% rotation error. This *is* a domain-shift stress test.
  - Per-scene statistics at scale: 96 ScanNet scenes, n=90 and n=65 valid subsets with bootstrap CIs
  - Scaling curves: 21 lengths × ScanNet, 12 × TUM, 10 × KITTI/Bonn — not single operating points
- **evidence gap**: no true "in-the-wild" (uncalibrated web/phone video) run. `examples/westlake.mp4`, `examples/taylor.mp4` + `demo.py` exist → **cheap qualitative in-the-wild demo is feasible**.
- **status**: open — needs decision on whether to run in-the-wild demo

### R1-C2 — Lack of rigorous theoretical analysis; Q2 theory for TTT3R's limitations
- **raw_anchor**: "more rigorous theoretical analysis would make the conclusions more solid"; Q2 "provide a theoretical analysis to demonstrate the existing limitations of TTT3R"
- **issue_type**: theorem_rigor · **severity**: major · **stance**: swing
- **response_mode**: assumption_hierarchy + grounded_evidence
- **evidence available**:
  - Appendix A.8 already derives DDD3R's effective magnitude √(α⊥²−(α⊥²−α∥²)cos²θ) and proves brake is its scalar relaxation
  - M2 already contains two *analytical* collapse arguments: TTT3R sigmoid saturation; TAUM's ‖Δs‖/mean(‖Δs‖) having mean 1.0 by construction ⇒ input centred at 1−τ ⇒ locks to sigmoid(−0.5)=0.378
  - `docs/theory_section.tex` (11 KB) exists but is **not currently included** in the paper
  - **A clean, provable proposition is available**: for any scalar gate g_t (however adaptive), S_t−S_{t−1} = g_t·δ_t is *parallel to δ_t*; hence the drift-energy fraction cos²(δ_t, d_t) of the applied update is **invariant to g_t**. Scalar gating therefore cannot alter directional composition — only DDD3R's two-coefficient form can. This is a one-line proof and is exactly the "why TTT3R is fundamentally limited" statement the reviewer asks for.
- **status**: open — proposition is derivable now, needs user sign-off as `user_confirmed_derivation`

### R1-C3 — Too many hyperparameters
- **raw_anchor**: "introduce a large number of hyperparameters, making it difficult for a single configuration to perform well across all datasets... encouraged to discuss possible solutions in the limitations section"
- **issue_type**: practical_significance · **severity**: major · **stance**: swing
- **response_mode**: direct_clarification + narrow_concession
- **evidence available**:
  - The **method** has 3 hyperparameters (α⊥, α∥, β_ema). The ~15 `auto_gamma` variants are *ablation instruments*, not part of the proposed method — a presentation problem, not a design problem.
  - Recommended default **DDD3R_brake has exactly ONE hyperparameter (τ)** and ranks 1st/2nd on every benchmark.
  - Sensitivity is flat and already measured: α∥ ∈ [0.05, 0.20] moves TUM ATE by <2%; α∥ ablation a10–a25 gives 0.055/0.055/0.056/0.061.
  - Honest concession: cross-dataset operating-point selection is unsolved (already in Limitations).
- **status**: open

---

## R2 — Reviewer DpBu

### R2-C1 — No cross-backbone validation (MASt3R, DUSt3R)
- **raw_anchor**: "does not validate the generality of DDD3R on other mainstream recurrent 3D reconstruction models such as MASt3R and DUSt3R"; Q1 "With this addition, the Quality score could be increased to 4"
- **issue_type**: baseline_comparison · **severity**: critical · **stance**: positive
- **response_mode**: direct_clarification + future_work_boundary (+ grounded_evidence if a run is feasible)
- **key clarification (respectful, verifiable)**: **DUSt3R and MASt3R are not recurrent** — they are pairwise/global-alignment feed-forward models with *no persistent state*, so there is no state update for DDD3R to regulate. Our own related work already says this: "cannot accumulate state" (`sec/related.tex`). Applying DDD3R to them is not possible by construction, not merely untested.
- **constructive counter-offer**: the meaningful cross-backbone test is another *recurrent* model (StreamVGGT / Spann3R / LONG3R / TTSA3R).
- **evidence gap**: **no second recurrent backbone in repo.** Requires new codebase + checkpoint + eval harness. Highest-cost item.
- **status**: needs_user_input — is a second recurrent backbone feasible in the rebuttal window?

### R2-C2 — "Complete absence of dynamic-scene evaluation"  ★ HIGHEST-VALUE ITEM
- **raw_anchor**: "All datasets used in the experiments consist of static indoor or outdoor scenes... In dynamic scenes, instantaneous object motion can substantially alter the drift vector, making the current fixed EMA tracking mechanism prone to failure."
- **issue_type**: empirical_support · **severity**: critical · **stance**: positive
- **response_mode**: direct_clarification + grounded_evidence
- **The premise is factually incorrect, and we can prove it with zero new runs:**
  - **TUM**: all 8 sequences are `freiburg3_sitting_*` / `freiburg3_walking_*` — precisely the TUM RGB-D **"Dynamic Objects"** category. `walking_*` has two people walking through and occluding large parts of the frame.
  - **Bonn**: the eval set is `rgbd_bonn_{balloon2, crowd2, crowd3, person_tracking2, synchronous}` — the **Bonn RGB-D *Dynamic*** dataset, dynamic by construction.
  - **Sintel**: animated sequences with large character/object motion; the standard dynamic benchmark in this line of work.
  - ⇒ **3 of 6 benchmarks are dynamic-scene benchmarks**, and TUM — source of the headline −67% — is *entirely* dynamic. The paper never labels them as such. **This is a presentation failure, not an experimental gap.**
- **NEW quantitative table, computed from existing raw eval output** (`eval_results/relpose/tum_s1_1000/*/…_eval_metric.txt`; all-8 mean reproduces the paper's 0.055 / 0.166 exactly):

  TUM 1000f ATE RMSE (m), split by dynamics severity:

  | Group | CUT3R | TTT3R | const | brake | **ortho** |
  |---|---|---|---|---|---|
  | `sitting_*` (mild dynamics, n=4) | 0.1424 | 0.0851 | 0.0624 | 0.0492 | **0.0407** |
  | `walking_*` (**severe** dynamics, n=4) | 0.1887 | 0.1199 | 0.0948 | 0.0776 | **0.0683** |
  | all 8 | 0.1656 | 0.1025 | 0.0786 | 0.0634 | **0.0545** |

  On **severe** dynamics: DDD3R_ortho **−63.8% vs CUT3R, −43.0% vs TTT3R**. DDD3R wins on **8/8** sequences.
  ⇒ Directly refutes "EMA tracking prone to failure": the margin under severe dynamics (−43.0% vs TTT3R) is
  close to that under mild dynamics (−52.1%) — degraded but far from failure, and still the best method.
- **Bonn (fully dynamic, 500f) abs_rel**, already in paper but never labelled dynamic:
  CUT3R 0.0819 · TTT3R 0.0720 · const 0.0687 · **brake 0.0661** · ortho 0.0678 → brake −19.3% vs CUT3R, −8.2% vs TTT3R.
- **status**: open — answerable *now*, no new experiments

### R2-C3 — No online γ selection algorithm
- **raw_anchor**: "does not design an automatic online γ selection algorithm without human intervention"; Q3 "Could the authors design a tuning-free mechanism"
- **issue_type**: practical_significance · **severity**: major · **stance**: positive
- **response_mode**: narrow_concession + grounded_evidence (negative results)
- **evidence available**: we ran **15+ online adaptive signals**, all failing to beat the fixed-config Pareto frontier — attention entropy, drift energy, local drift energy (6 variants), drift growth, projection fraction, momentum resultant, frame-mean sigmoid, warmup-linear/threshold, steep sigmoid/clamp, drift-confidence (3 variants), ortho+brake composition. TUM/ScanNet numbers exist for all (`docs/experiment_results.md`).
  e.g. TUM: entropy 0.070, drift_energy 0.057, drift_growth 0.056, proj_frac 0.062, momentum 0.064 — none beats fixed ortho 0.055. ScanNet: fmean_sig 0.279, local_de_raw 0.284, entropy 0.294 — none beats brake 0.261.
- **framing**: this is a *systematic negative result* worth reporting, not an omission. It converts "you didn't try" into "we exhausted the online-signal space and this is an open problem" — and the paper's stated contribution is diagnostic, not a claim to have solved selection.
- **status**: open — answerable now

---

## R3 — Reviewer 1ake  ★ LOWEST QUALITY SCORE, EXPLICIT PROMISE TO RAISE

### R3-C1 — Missing positioning vs token-compression / state-bottleneck work
- **raw_anchor**: "does not engage with... ZPressor and iLRM. Both are orthogonal in mechanism but tackle a related symptom. You can refer to the [3] survey"
- refs given: [1] ZPressor (NeurIPS 2025), [2] Long-LRM (ICCV 2025), [3] Feed-Forward 3D Scene Modeling survey (arXiv 2026)
- **issue_type**: novelty / clarity · **severity**: major · **stance**: swing
- **response_mode**: nearest_work_delta
- **cost**: **pure writing** — one related-work paragraph. Cheapest possible fix for the lowest scorer.
- **delta to articulate**: ZPressor / Long-LRM reduce redundancy in the *spatial/token* dimension (compressing how many tokens carry the state) — a capacity bottleneck. DDD3R addresses redundancy in the *temporal-directional* dimension (what direction each update moves the state). Orthogonal and composable: compression changes the state's width, DDD3R changes its update trajectory.
- **BLOCKER**: all three citations must be verified via DBLP/CrossRef before inclusion. Reviewer wrote "iLRM" in prose but cited "Long-LRM" — need to determine whether these are two distinct papers.
- **status**: open — citation verification required

### R3-C2 — TTSA3R/TAUM has no head-to-head empirical row
- **raw_anchor**: "TTSA3R / TAUM is discussed analytically (M2) but is not given a head-to-head row in Tables 3–4 on long-sequence benchmarks, leaving the prediction 'TAUM ≡ constant dampening' empirically unconfirmed"
- **issue_type**: baseline_comparison · **severity**: critical · **stance**: swing
- **response_mode**: grounded_evidence
- **evidence available**:
  - TAUM gate statistics **already measured** end-to-end: `analysis/taum_gate_stats.py` with `cut3r_taum_log` update type → σ_time = 0.006, μ ≈ 0.355 vs predicted sigmoid(−0.5) = 0.378. The *collapse claim* is empirically confirmed.
  - Closest existing ATE proxy: `ttt3r_l2gate` (L2-norm gate, same family) = 0.075 TUM 1000f, i.e. between TTT3R (0.103) and constant (0.079) — consistent with "≡ constant dampening".
- **evidence gap**: no end-to-end **ATE** run of full TAUM×SCUM as an update type. `cut3r_taum_log` already implements the TAUM formula for logging ⇒ **converting it into a live update type is a small code change**, then 1× TUM + 1× ScanNet 1000f eval.
- **assessment**: **highest ROI new experiment in the whole rebuttal** — cheap, directly targets the lowest scorer's critical concern, and confirms a prediction the paper already makes.
- **status**: needs_user_input — approve TAUM head-to-head run?

### R3-C3 — Single backbone family
- **raw_anchor**: "All gains are reported on one CUT3R/TTT3R checkpoint... unclear whether [this] is a property of recurrent 3D reconstruction in general or of this specific checkpoint"
- duplicate of **R2-C1** — answer once, cross-reference.
- **status**: merged into R2-C1

---

## Coverage Summary

| ID | Severity | Answerable now? | Needs new work |
|---|---|---|---|
| R1-C1 real-world | major | partly (KITTI OOD framing) | optional in-the-wild demo |
| R1-C2 theory | major | yes (proposition + A.8) | user sign-off on derivation |
| R1-C3 hyperparams | major | **yes** | — |
| R2-C1 cross-backbone | critical | partly (DUSt3R/MASt3R not recurrent) | 2nd recurrent backbone (expensive) |
| R2-C2 dynamic scenes | critical | **yes — fully** | — |
| R2-C3 online γ | major | **yes** (negative results) | — |
| R3-C1 related work | major | yes | citation verification |
| R3-C2 TAUM row | critical | partly (gate stats) | TAUM ATE run (cheap, high ROI) |
| R3-C3 single backbone | major | → R2-C1 | — |

**5 of 9 fully answerable with zero new experiments.** Two cheap adds (TAUM row, in-the-wild demo)
and one expensive add (second recurrent backbone) would close the rest.
