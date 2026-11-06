# REBUTTAL DRAFT v1 — Submission 8591 (DDD3R)

Venue: NeurIPS 2026 · Format: text-only · **Limit: 6000 characters per reviewer**

Status of new experiments — **all complete, no `[[PENDING]]` markers remain**:
- **E1 TTSA3R head-to-head** ✅ — full row on both main tables (relpose 5 settings + video depth 3)
- **E3 StreamVGGT cross-backbone** ✅ — architectural finding + measured 270 MB/frame cache growth
- **E4 ARKitScenes** ✅ — pre-registered predictive test, 18 scenes × 500 f, real-world iPad capture
  (drift energy locked in commit `b74c4c0` *before* any ATE; results in `fd48bf0`)
- **E2 in-the-wild qualitative demo** — optional, not run; no claim in the draft depends on it

---

## → Reviewer 1ake

We thank you for the precise, actionable review. All three concerns are addressed with new
material; two required no change of claim, one required a new experiment.

**W1 — Positioning vs. token-compression / state-bottleneck work.**
You are right that this line belongs in our related work, and we will add a paragraph citing
ZPressor [NeurIPS'25], Long-LRM [ICCV'25], iLRM [arXiv'25] and the survey you point to. (We note
your text mentions *iLRM* while ref. [2] is *Long-LRM* — these are distinct works, so we cite both.)
The distinction we draw is axis-of-redundancy: ZPressor/Long-LRM/iLRM reduce redundancy along the
**spatial-token / capacity** axis — how many tokens carry the state, i.e. compressing its *width* —
via information-bottleneck compression, token merging, or iterative refinement over a decoupled
representation. DDD3R targets redundancy along the **temporal-directional** axis — which direction
each update moves the state, i.e. regulating its *trajectory*. These are orthogonal and composable:
a compressed state still accumulates drift across frames, and DDD3R would regulate it identically.
We think this framing strengthens rather than dilutes our positioning, and we thank you for it.

**W2 — TTSA3R/TAUM has no head-to-head row.**
This was a fair gap and we have now closed it experimentally. We implemented TTSA3R's TAUM×SCUM gate
as a live update rule (previously we only *logged* what the gate would output) and verified it
line-by-line against the authors' released code; every operation — state-change normalization,
sigmoid(·−1.5), feature dissimilarity, cross-attention magnitude, spatial max-fusion — matches
exactly. Results, same protocol as Tables 3–4:

**Relative pose (ATE, m ↓)**

| Method | TUM 90f | TUM 1000f | ScanNet 90f | ScanNet 1000f | Sintel |
|---|---|---|---|---|---|
| CUT3R | 0.033 | 0.166 | 0.095 | 0.817 | 0.209 |
| TTT3R | 0.019 | 0.103 | 0.064 | 0.406 | 0.209 |
| **TTSA3R** | 0.016 | 0.091 | **0.058** | 0.378 | 0.209 |
| DDD3R_const | 0.016 | 0.079 | 0.065 | 0.283 | 0.220 |
| DDD3R_brake | 0.015 | 0.063 | 0.072 | **0.261** | 0.237 |
| DDD3R_ortho | **0.015** | **0.055** | 0.087 | 0.488 | 0.236 |

**Video depth (abs_rel ↓)**: TTSA3R 0.106 / 0.072 / 0.425 on KITTI / Bonn / Sintel, vs
CUT3R 0.119 / 0.082 / 0.467 and TTT3R 0.107 / 0.072 / 0.433.

We report this precisely, including where it does **not** match our prediction. TTSA3R improves on
TTT3R in every cell, and on ScanNet 90f it is the **best method overall** (0.058), ahead of every
DDD3R operating point. So the blanket reading "TAUM ≡ constant dampening" is **too strong, and we
will soften that wording**. The likely reason is that TTSA3R fuses two gates, TAUM (temporal) *and*
SCUM (spatial); our M2 analysis concerns the **temporal** one, and we measure its collapse directly
(σ_temporal ≈ 0.006, locking near sigmoid(−0.5) = 0.378). SCUM's spatial selectivity is a different
mechanism and evidently helps at short horizons.

What the long-sequence regime does confirm is the substantive claim: a scalar gate, however well
designed, plateaus once over-update accumulates. At 1000 frames TTSA3R trails DDD3R_ortho by
**39.9%** on TUM (0.091 vs 0.055) and DDD3R_brake by **30.9%** on ScanNet (0.378 vs 0.261), and on
Sintel (~50 f) all three gates are indistinguishable (0.209). TTSA3R is a better gate than TTT3R's;
it is still only a gate, and cannot alter the directional composition of the update — which is
exactly the invariance we prove in our reply to Reviewer ULz9.

**W3 — Dependence on a single backbone family.**
We ran a cross-backbone study on **Point3R** [NeurIPS'25], a streaming 3D reconstruction model with
a *different backbone* (DUSt3R ViT-L, not CUT3R's ARCroco3DStereo) and a *different memory design*
(an explicit set of spatial pointers, each anchored to a 3D position and growing with the scene,
rather than a fixed 768-token implicit state). Its merge step fully replaces a pointer's feature
with new content (`memory_feat[idx] = feat_avg`, i.e. β=1, no dampening) — the extreme of our M1.
We changed only that line to `mem ← mem + α(new − mem)`; α=1 reproduces upstream bit-exactly and
serves as the baseline.

The diagnosis **splits**, and both halves are informative.
*M1 transfers.* Instrumenting the merge, updates have norm ≈28% of the stored feature with no
dampening. Applying our fix: on ScanNet (n=90) α=0.5 gives **−5.4%** ATE (Wilcoxon p=0.0022, 95% CI
[−0.0083,−0.0024], better on 56/90 scenes); on TUM (n=8) −20.1%, though n=8 cannot certify that
(p=0.250). Both datasets select **α=0.5 — the same coefficient we use on CUT3R.**
*M3 does not transfer.* Drift energy is 0.155–0.170 across 7 sequences and two datasets, versus
0.398 (TUM) / 0.598 (ScanNet) for CUT3R; on **identical ScanNet scenes** it falls 0.598 → 0.164, and
cos(δ_t,δ_{t−1}) is *negative*. Anchoring each pointer to a 3D position means it is updated by
different viewpoints at different times, which decorrelates the update direction.

So over-update is architecture-independent, while directional redundancy is specific to a bounded,
repeatedly-overwritten *implicit* state; Point3R avoids it by letting memory grow with the scene, at
the cost of bounded memory. This is also a successful prediction of C3 — low drift energy correctly
implies directional decomposition should not be used there. We additionally checked StreamVGGT
[ICLR'26] and Spann3R: both are append-only (`torch.cat`; Spann3R bounds memory by top-k eviction,
never editing a written token), so no Δ_t exists and DDD3R is undefined rather than ineffective.
StreamVGGT's cache grows a measured 270 MB/frame (~270 GB at 1000 frames) against DDD3R's constant
1.18 MB — the memory-vs-drift trade-off that defines the family we target.

---

## → Reviewer DpBu

Thank you for the detailed and generous reading. One weakness rests on a premise we ourselves caused
by under-describing our datasets, and we are grateful for the chance to correct it.

**W2 — "Complete absence of dynamic-scene evaluation."**
We must apologise: our appendix never labels the dataset characteristics, so this conclusion was
entirely reasonable. In fact **three of our six benchmarks are dynamic-scene benchmarks**:

- **TUM**: all 8 sequences are `freiburg3_sitting_*` / `freiburg3_walking_*` — precisely TUM RGB-D's
  **"Dynamic Objects"** category. The `walking_*` sequences contain two people walking through and
  occluding large portions of the frame.
- **Bonn**: we use `rgbd_bonn_{balloon2, crowd2, crowd3, person_tracking2, synchronous}` — the
  **Bonn RGB-D *Dynamic*** dataset, dynamic by construction.
- **Sintel**: animated sequences with large character and object motion.

So our headline TUM result (−67%) is *already* obtained entirely on dynamic sequences. To address
your concern directly — that instantaneous object motion could break EMA drift tracking — we split
TUM 1000f by dynamics severity (ATE RMSE, m):

| Group | CUT3R | TTT3R | const | brake | **ortho** |
|---|---|---|---|---|---|
| `sitting_*` (mild, n=4) | 0.1424 | 0.0851 | 0.0624 | 0.0492 | **0.0407** |
| `walking_*` (**severe**, n=4) | 0.1887 | 0.1199 | 0.0948 | 0.0776 | **0.0683** |

Under **severe** dynamics DDD3R still gives **−63.8% vs CUT3R and −43.0% vs TTT3R**, and wins on
**8/8** sequences. The margin narrows from −52.1% (mild) to −43.0% (severe) — measurable degradation,
but far from the failure mode you hypothesise. On fully-dynamic Bonn (500f) every DDD3R variant beats
both baselines (brake 0.0661 vs CUT3R 0.0819, −19.3%). We will label dataset dynamics explicitly and
add this table.

**W1/Q1 — Cross-backbone validation on MASt3R and DUSt3R.**
Respectfully, DUSt3R and MASt3R are **not recurrent**: they are pairwise/global-alignment feed-forward
models with no persistent state carried across frames, so there is no state update for DDD3R to
regulate — inapplicable by construction rather than untested (our related work states this as
"cannot accumulate state"). We agree the meaningful test is a *different recurrent* model, and we ran
one: **Point3R** [NeurIPS'25], which uses the DUSt3R ViT-L backbone with an explicit spatial-pointer
memory instead of CUT3R's implicit 768-token state. Its merge overwrites a pointer outright
(β=1); changing that single line to `mem ← mem + α(new − mem)` (α=1 reproduces upstream exactly)
gives **−5.4% ATE on ScanNet, n=90, p=0.0022** at **α=0.5, the same coefficient we use on CUT3R**
(TUM n=8: −20.1%, not significant at that sample size). So **M1 transfers across backbones**.
Interestingly **M3 does not**: Point3R's drift energy is 0.164 on the *same* ScanNet scenes where
CUT3R's is 0.598, because anchoring each pointer to a 3D position means different viewpoints update
it at different times, decorrelating the update direction. Point3R avoids the directional pathology
by letting memory grow with the scene — trading away bounded memory to do so. Full details, plus
StreamVGGT and Spann3R (both append-only, hence no Δ_t to regulate), are in our reply to Reviewer
1ake (W3). We will add this cross-backbone study and the resulting scope statement to the paper.

**W3/Q3 — Online γ selection.**
We concede this fully; it is the paper's main open problem. We report it as a systematic negative
result rather than an omission: we evaluated **15+ online signals** — attention entropy, drift energy,
local drift energy (6 variants), drift growth, projection fraction, momentum resultant, frame-mean
sigmoid, warmup-linear/threshold, steep-sigmoid/clamp, drift-confidence (3 variants) — and **none**
beat the fixed-configuration Pareto frontier (TUM: entropy 0.070, drift-energy 0.057, proj-frac 0.062
vs fixed ortho 0.055; ScanNet: fmean-sig 0.279, entropy 0.294 vs brake 0.261). We will add this
sweep to the appendix, as evidence that the difficulty is intrinsic rather than unexplored.

---

## → Reviewer ULz9

Thank you — we address the theory request with a formal result and the hyperparameter concern with a
correction of our own presentation.

**Q2/W2 — Theoretical analysis of TTT3R's limitation.**
We can state the limitation as a short invariance result. Let δ_t be the raw per-token delta and d_t
the tracked drift direction. Any scalar (per-token) gate g_t applied as S_t = S_{t−1} + g_t·δ_t yields
an applied update u_t = g_t·δ_t, and for every g_t ≠ 0:

  **cos²(u_t, d_t) = cos²(δ_t, d_t).**

The drift-energy fraction — precisely the quantity M3 identifies as harmful — is therefore
**invariant to the gate**. A scalar gate can only rescale ‖u_t‖; it cannot alter the drift/novelty
ratio, no matter how adaptive it is. Changing that ratio requires at least two coefficients, which is
exactly Eq. (5).

**Corollary.** TTT3R's attention gate, TTSA3R's TAUM×SCUM, and constant dampening all lie in the
same equivalence class *with respect to directional composition*; they can differ only in their
magnitude schedule. This is a statement about what scalar gating **cannot** do, not a claim that all
scalar gates perform alike — magnitude schedules do matter, and our new TTSA3R evaluation (reply to
Reviewer 1ake) shows its spatial gating genuinely helps at short horizons. The point is that no
magnitude schedule, however sophisticated, addresses M3; and empirically all of them converge to the
same plateau once over-update accumulates on long sequences. This complements the existing
Appendix A.8 derivation, and we will add it as a formal subsection in the camera-ready.

**W1/Q1 — Real-world generalization.** We treat these as one concern and answer it with a new
experiment rather than an argument.

We evaluated on **ARKitScenes** — 18 real-world scenes, 500 frames each, captured handheld on a
consumer iPad Pro, with ground-truth poses from registered Faro laser scans (independent of any
visual SfM). Scene selection was fixed in advance (Validation split, video_id ascending, first 20;
two scenes were too short for 500 frames).

Crucially, we ran this as a **pre-registered predictive test of our own theory, not a search for a
win**. We first measured drift energy — a property of the data and model, computable without any
pose evaluation — obtaining **ē = 0.632 ± 0.035**, i.e. *higher* than ScanNet (0.598) and far from
TUM (0.398). We then committed, in a timestamped record before computing any ATE, to the prediction
that C3 implies: on such a high-drift dataset **isotropic dampening should win and full directional
decomposition should lose** — that is, our own flagship \dddortho{} should rank *worst* of the three
DDD3R operating points.

Result (ATE, m ↓): CUT3R 0.682 · TTT3R 0.558 · TTSA3R 0.556 · **\dddconst{} 0.528** ·
\dddbrake{} 0.539 · \dddortho{} 0.609.

The prediction is confirmed on the point that discriminates it: \dddortho{} is indeed the worst
DDD3R variant (0.609, vs 0.528 for \dddconst{} and 0.539 for \dddbrake{}; Wilcoxon p=0.027 with
14/18 scene wins, and p=0.081 with 12/18, respectively). \dddconst{} improves on CUT3R by 22.6%
(p=0.001) and on TTT3R by 5.5% (p=0.027).

We also report where the prediction **failed**, since that is the point of pre-registering it.
(i) We predicted the strict order brake < const < ortho; const and brake in fact swapped. They
differ by 2.2% at p=0.671, so we claim no ordering between them. (ii) We predicted *every* DDD3R
point would beat both baselines; **\dddortho{} is 9.2% worse than TTT3R here** (4/18 scenes). This
is a genuine limitation, and it is the same boundary already visible in our paper on ScanNet 1000f,
the other high-drift dataset, where \dddortho{} (0.488) also trails TTT3R (0.406). We will narrow
the recommended scope of \dddortho{} to low-drift-energy scenes accordingly.

We think this is a stronger answer to your question than another benchmark win would have been: on
data we had never touched, a scene-intrinsic quantity measured *before* any evaluation correctly
predicted which configuration would fail. Alongside this, our existing evaluation already contains a
domain-shift test we under-framed: **KITTI Odometry** (outdoor driving, 11 sequences, 271–4661
frames) is out-of-domain for an indoor-trained checkpoint, and \dddortho{} improves ATE there by
20.1% and rotation error by 58.6%.

**W3 — Too many hyperparameters.**
This is our presentation's fault and we will fix it. The **method** has three hyperparameters
(α⊥, α∥, β_ema); the ~15 variants in the appendix are *diagnostic instruments* used to map the
spectrum, not knobs a user must set. The configurations we actually recommend are far simpler:
\dddbrake{} has **exactly one** (τ) and \dddconst{} **exactly one** (α). Sensitivity is flat —
α∥ ∈ [0.05, 0.20] moves TUM ATE by <2% (0.055/0.055/0.055/0.056) — so performance does not hinge on
careful tuning.

The honest residual difficulty is not the *number* of hyperparameters but *which operating point* to
pick, and our new ARKitScenes experiment sharpens rather than hides this. The isotropic points are
the robust default: \dddconst{} is best there (0.528, −22.6% vs CUT3R, p=0.001) and \dddbrake{}
second, while \dddortho{} degrades — consistent with the high drift energy we measured in advance.
Combined with the paper's existing results this gives a concrete rule: **use an isotropic point
(\dddconst{}/\dddbrake{}) unless the scene is known to be low-drift (ē ≲ 0.45, TUM-like), where
\dddortho{} adds a further ~13%.** We will restructure Sec. 4 to foreground the three-parameter
method, move the spectrum instruments to the appendix, and expand Limitations as you suggest to
state that automatic cross-regime selection remains open (see also our reply to DpBu W3).
