We thank the reviewer for a precise and actionable review. All three points are addressed with new material, and two required new experiments.

---

**W1 — Positioning against token-compression and state-bottleneck work.**

**(1) We agree.** This line belongs in our related work and we will add a paragraph citing ZPressor (NeurIPS 2025), Long-LRM (ICCV 2025), iLRM (arXiv 2025) and the survey you point to. We note that your text mentions *iLRM* while ref. [2] is *Long-LRM*. These are distinct works by different groups, so we cite both.

**(2) The distinction we draw is the axis of redundancy.** ZPressor, Long-LRM and iLRM reduce redundancy along the **spatial-token or capacity** axis, that is, how many tokens carry the state, compressing its *width*, via information-bottleneck compression, token merging, or iterative refinement over a decoupled representation. DDD3R targets the **temporal-directional** axis, that is, which direction each update moves the state, regulating its *trajectory*.

**(3) The two are orthogonal and composable.** A compressed state still accumulates drift across frames, and DDD3R would apply to it unchanged. We see this framing as strengthening rather than diluting our positioning, and we thank you for raising it.

---

**W2 — TTSA3R and TAUM have no head-to-head row.**

**(1) What we did.** A fair gap. Following the authors' released code, we implement TTSA3R's TAUM$\times$SCUM gate as a live update rule, where previously we only logged what the gate would output.

**(2) Relative pose**, ATE in m $\downarrow$, same protocol as Table 3.

| Method | TUM 90f | TUM 1000f | ScanNet 90f | ScanNet 1000f | Sintel |
|---|---|---|---|---|---|
| CUT3R | 0.032 | 0.166 | 0.095 | 0.817 | 0.209 |
| TTT3R | 0.019 | 0.103 | 0.064 | 0.406 | 0.209 |
| **TTSA3R** | 0.016 | 0.091 | **0.058** | 0.378 | 0.209 |
| DDD3R-const | 0.016 | 0.079 | 0.065 | 0.283 | 0.220 |
| DDD3R-brake | 0.015 | 0.063 | 0.072 | **0.261** | 0.237 |
| DDD3R-ortho | **0.015** | **0.055** | 0.087 | 0.488 | 0.236 |

**(3) Video depth**, abs_rel $\downarrow$, same protocol as Table 4.

| Method | KITTI | Bonn | Sintel |
|---|---|---|---|
| CUT3R | 0.119 | 0.082 | 0.465 |
| TTT3R | 0.107 | 0.072 | 0.433 |
| **TTSA3R** | 0.106 | 0.072 | 0.425 |
| DDD3R-brake | 0.106 | **0.066** | **0.402** |
| DDD3R-ortho | **0.103** | 0.068 | 0.418 |

**(4) What this confirms, and where our wording was too strong.** TTSA3R improves on TTT3R in every cell, and on ScanNet 90f it is the best method overall at $0.058$. The blanket reading that TAUM reduces to constant dampening is therefore **too strong and we will soften it**. The likely reason is that TTSA3R fuses two gates, TAUM (temporal) and SCUM (spatial). Our M2 analysis concerns the temporal one, whose collapse we measure directly at $\sigma_{\text{temporal}} \approx 0.006$, while SCUM's spatial selectivity is a different mechanism that evidently helps at short horizons.

**(5) What the long-sequence regime does confirm.** A scalar gate, however well designed, plateaus once over-update accumulates. At 1000 frames TTSA3R trails DDD3R-ortho by $39.9\%$ on TUM and DDD3R-brake by $30.9\%$ on ScanNet, and on Sintel at roughly 50 frames all three gates are indistinguishable at $0.209$. TTSA3R is a better gate than TTT3R's, but it remains a gate and cannot alter the directional composition of the update, which is the invariance we prove in our reply to Reviewer ULz9.

---

**W3 — Dependence on a single backbone family.**

**(1) A different recurrent backbone.** We ran **Point3R** (NeurIPS 2025), which uses the DUSt3R ViT-L backbone with an explicit spatial pointer memory rather than CUT3R's fixed 768-token implicit state. Its merge overwrites a pointer feature outright, which is $\beta = 1$ with no dampening. Changing only that line to $\mathrm{mem} \leftarrow \mathrm{mem} + \alpha(\mathrm{new} - \mathrm{mem})$, where $\alpha = 1$ reproduces the released model exactly, gives $-5.4\%$ ATE on ScanNet over 90 scenes (Wilcoxon $p=0.0022$) at $\alpha = 0.5$, the same coefficient we use on CUT3R.

**(2) M1 transfers, M3 does not.** Point3R's drift energy is $0.164$ on the *same* ScanNet scenes where CUT3R's is $0.598$, and $\cos(\delta_t, \delta_{t-1})$ is negative. Anchoring each pointer to a 3D position means different viewpoints update it at different times, which decorrelates the update direction. So over-update is architecture-independent, while directional redundancy is specific to a bounded, repeatedly overwritten implicit state. Point3R avoids that pathology by letting memory grow with the scene, at the cost of bounded memory.

**(3) Other candidates.** We also inspected StreamVGGT (ICLR 2026) and Spann3R. Both are append-only, so neither has a $\Delta_t$ to regulate and DDD3R is undefined there rather than ineffective. Fuller detail is in our reply to Reviewer DpBu (W1).

We will add the TTSA3R rows, the cross-backbone study and the resulting scope statement to the paper.
