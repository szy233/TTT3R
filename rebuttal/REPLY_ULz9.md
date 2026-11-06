We thank the reviewer for the questions, and respond point by point below.

---

**W1 / Q1 — Real-world generalization.**

**(1) Setup.** We evaluate on **ARKitScenes**: 18 real-world indoor scenes, 500 frames each, captured handheld on a consumer iPad Pro, with ground truth from registered Faro laser scans (independent of visual SfM).

**(2) Drift analysis, run first.** Drift energy depends only on data and model, so we measured it before any pose evaluation, obtaining $\bar{e} = 0.632 \pm 0.035$. This is above ScanNet ($0.598$) and far from TUM ($0.398$), i.e. a high-drift regime. Under our claim that *drift energy, a scene-intrinsic property, governs the appropriate operating point* (Sec. 3.3, App. A.4), this predicts that isotropic dampening wins and directional decomposition loses, so our own default DDD3R-ortho should rank worst. We record that prediction before computing any ATE.

**(3) Results** (ATE, m $\downarrow$):

| CUT3R | TTT3R | TTSA3R | **DDD3R-const** | DDD3R-brake | DDD3R-ortho |
|---|---|---|---|---|---|
| 0.682 | 0.558 | 0.556 | **0.528** | 0.539 | 0.609 |

DDD3R-const: $-22.6\%$ vs CUT3R ($p=0.001$), $-5.5\%$ vs TTT3R ($p=0.027$). Ortho is the worst DDD3R variant, behind const ($p=0.027$, 14/18 scenes).

**(4) Why.** The prediction holds because here drift carries useful geometric refinement rather than harmful repetition, so suppressing it directionally removes signal. The same boundary is already visible in the paper on ScanNet 1000f, our other high-drift dataset, where ortho ($0.488$) likewise trails TTT3R ($0.406$). We will narrow DDD3R-ortho's scope to low-drift scenes, and recommend the isotropic configurations (DDD3R-const / DDD3R-brake) everywhere else.

**(5) Outdoor / domain shift.** Our evaluation already includes one, **KITTI Odometry** (11 sequences, 271–4661 frames; Table 3), where ortho cuts ATE by $20.1\%$. The checkpoint is trained on short indoor-heavy pairs, while KITTI is outdoor driving up to 4661 frames. We concede we never framed it as a domain-shift test. On the official KITTI metrics ortho additionally reduces translation error by $7.6\%$ ($93.94 \to 86.77$) and rotation error by $58.6\%$ ($22.66 \to 9.38$ deg/100m).

---

**W2 / Q2 — Theoretical analysis of TTT3R's limitation.**

TTT3R has two limitations.

**(1) The gate collapses.** This is empirical and already in the paper. Saturated cross-attention scores keep the frame-averaged gate within $\mu \in [0.31, 0.35]$ at $\sigma_{\text{temporal}} \le 0.03$, across TUM, ScanNet and KITTI (Sec. 3.2).

**(2) Even a non-collapsing gate would not suffice.** For any scalar per-token gate $g_t$, the applied update $u_t = g_t \delta_t$ is collinear with the raw delta $\delta_t$, so for every $g_t \neq 0$

$$\cos^2(u_t, d_t) = \cos^2(\delta_t, d_t).$$

The drift-energy fraction, the quantity M3 identifies as harmful, is therefore **invariant to the gate**. The argument holds for *any* fixed reference direction $d_t$, so this is a property of scalar gating itself rather than of our EMA construction. A scalar gate rescales $\|u_t\|$ only. No schedule, however adaptive, alters the drift-to-novelty ratio. Two coefficients are the minimum that can: under Eq. (5) the applied drift fraction becomes $\alpha_\parallel^2 \|\delta_\parallel\|^2 / (\alpha_\perp^2 \|\delta_\perp\|^2 + \alpha_\parallel^2 \|\delta_\parallel\|^2)$, which collapses back to the invariant case if $\alpha_\perp = \alpha_\parallel$.

**Corollary.** TTT3R's gate, TTSA3R's TAUM$\times$SCUM and constant dampening form one equivalence class for directional composition, differing only in magnitude schedule. This states what scalar gating cannot do. It is not a claim that all scalar gates perform alike: our TTSA3R evaluation (reply to Reviewer 1ake) shows its spatial term does help at short horizons.

---

**W3 — Too many hyperparameters.**

This is a presentation failure on our part. DDD3R has four hyperparameters ($\alpha_\perp$, $\alpha_\parallel$, $\beta_{\text{ema}}$, $\gamma$), but $\gamma$ is not one a user is expected to tune. The paper already fixes $\gamma = 0$ in the default configuration $(0.5, 0.05, 0.95, 0)$ and describes the drift-adaptive mode $\gamma > 0$ as an ablation (App. A.1) and as a spectrum-exploration tool rather than a recommended setting (Sec. 4). The $\gamma$ sweep in Table 5 is a diagnostic instrument used to map the spectrum, not a menu of configurations to choose from. In practice a user sets three coefficients, and our recommended operating points reduce to one. Sensitivity is also flat: on TUM, $\alpha_\parallel \in [0.05, 0.20]$ moves ATE by $<2\%$ ($0.055/0.055/0.055/0.056$), degrading only past $\alpha_\parallel = 0.25$ ($0.061$), and $\beta_{\text{ema}} = 0.95$ is consistently optimal (App. A.1, Sec. 5.2).

**Limitations.** As you encourage, we will expand the Limitations paragraph. The real difficulty is not the parameter count but which operating point to pick, and we now have a better account of why the adaptive variants fail: per-frame signals (entropy, drift growth, momentum, proj-frac) saturate because the right operating point is a sequence-level property rather than a per-frame one. The ARKitScenes result above points to a feasible alternative, namely estimating drift energy on a short prefix and then fixing the operating point for the rest of the sequence.
