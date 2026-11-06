We thank the reviewer for the detailed reading, and respond point by point below.

---

**W1 / Q1 — Cross-backbone validation.**

**(1) Why not MASt3R / DUSt3R.** Neither is recurrent. They are pairwise or global-alignment feed-forward models with no persistent state carried across frames, so there is no state update for DDD3R to regulate. It is inapplicable by construction rather than untested, as our related work states with "cannot accumulate state".

**(2) What we ran instead.** We agree the meaningful test is a different *recurrent* model, so we ran **Point3R** (NeurIPS 2025). It uses the DUSt3R ViT-L backbone with an **explicit spatial pointer memory**, each pointer anchored to a 3D position and the set growing with the scene, rather than CUT3R's fixed 768-token implicit state. Its merge replaces a pointer feature outright, `memory[idx] = feat_avg`, i.e. $\beta = 1$ with no dampening, the extreme of our M1. We changed only that line to $\mathrm{mem} \leftarrow \mathrm{mem} + \alpha(\mathrm{new} - \mathrm{mem})$, where $\alpha = 1$ reproduces the released model bit-exactly and serves as the baseline.

**(3) Results**, ATE in m $\downarrow$.

| | upstream ($\alpha{=}1$) | $\alpha{=}0.7$ | $\alpha{=}0.5$ |
|---|---|---|---|
| ScanNet ($n{=}90$) | 0.0971 | 0.0948 | **0.0918** |
| TUM ($n{=}8$) | 0.0427 | 0.0382 | **0.0341** |

**M1 transfers.** On ScanNet $\alpha = 0.5$ gives $-5.4\%$ (Wilcoxon $p=0.0022$, better on 56/90 scenes). On TUM the effect is larger at $-20.1\%$ but $n=8$ is too small to certify it. Both datasets select $\alpha = 0.5$, the same coefficient we use on CUT3R.

**(4) What does not transfer.** Point3R's drift energy is $0.164$ on the *same* ScanNet scenes where CUT3R's is $0.598$, and $\cos(\delta_t, \delta_{t-1})$ is negative. Anchoring each pointer to a 3D position means different viewpoints update it at different times, which decorrelates the update direction. Point3R avoids the directional pathology by letting memory grow with the scene, at the cost of bounded memory. So M1 is architecture-independent while M3 is specific to a bounded, repeatedly overwritten implicit state.

**(5) Other candidates.** We also inspected **StreamVGGT** (ICLR 2026) and **Spann3R**. Both are append-only, so neither has a $\Delta_t$ to regulate and DDD3R is undefined there rather than ineffective.

---

**W2 / Q2 — Dynamic-scene evaluation.**

**(1) The premise, which our own writing caused.** Our appendix never labels dataset characteristics, so this conclusion was entirely reasonable. In fact **three of our six benchmarks are dynamic-scene benchmarks**. All 8 TUM sequences are `freiburg3_sitting_*` and `freiburg3_walking_*`, precisely TUM RGB-D's *Dynamic Objects* category, where `walking_*` has two people walking through and occluding large parts of the frame. On Bonn we use `rgbd_bonn_{balloon2, crowd2, crowd3, person_tracking2, synchronous}`, the Bonn RGB-D *Dynamic* dataset. Sintel consists of animated sequences with large character motion. Our headline TUM result is therefore already obtained entirely on dynamic sequences.

**(2) Severity split.** To address your concern directly, that instantaneous object motion could break EMA drift tracking, we split TUM 1000f by dynamics severity, ATE in m $\downarrow$.

| Group | CUT3R | TTT3R | const | brake | **ortho** |
|---|---|---|---|---|---|
| `sitting_*` (mild, $n{=}4$) | 0.142 | 0.085 | 0.062 | 0.049 | **0.041** |
| `walking_*` (severe, $n{=}4$) | 0.189 | 0.120 | 0.095 | 0.078 | **0.068** |

Under severe dynamics DDD3R still gives $-63.8\%$ over CUT3R and $-43.0\%$ over TTT3R, and is best on **8/8** sequences. On the fully dynamic Bonn (500f) every variant beats both baselines, brake reaching $0.066$ against CUT3R's $0.082$.

**(3) Why we propose no separate mechanism.** The margin narrows from $-52.1\%$ (mild) to $-43.0\%$ (severe), so dynamics do degrade EMA tracking measurably, but far less than a failure. We would rather report that than add a component the evidence does not call for. We will label dataset characteristics explicitly and add this split to the paper.

---

**W3 / Q3 — Online $\gamma$ selection.**

**(1) $\gamma$ is not a user parameter.** The default configuration fixes $\gamma = 0$, and the drift-adaptive mode $\gamma > 0$ is described as an ablation (App. A.1) and as a spectrum-exploration tool rather than a recommended setting (Sec. 4).

**(2) What fails, and why.** We evaluated per-frame adaptive signals (attention entropy, drift energy, drift growth, projection fraction, momentum) and none improved on a fixed configuration. We now attribute this to a mismatch of timescales. The quantity determining the right operating point is a **sequence-level** property of the scene, whereas these signals adapt every frame and mostly track local noise.

**(3) A feasible direction.** Our ARKitScenes experiment (reply to Reviewer ULz9) shows drift energy is measurable from data and model alone, before any evaluation, and a single up-front estimate correctly predicted which configuration would fail on a dataset we had never touched. A deployable rule would estimate drift energy on a short prefix and then **fix** the operating point for the remainder. Establishing the prefix length, a robust threshold, and behaviour under regime changes within one sequence is concrete future work.

---

**On the Limitations discussion.** You correctly note our limitations were qualitative. The experiments above now quantify all three cases you name. Cross-backbone degradation is $-5.4\%$ on Point3R, with M3 shown not to transfer and the reason identified. Dynamic scenes are covered by the severity split above. For ultra-long sequences the paper reports scaling curves over 21 lengths on ScanNet and 12 on TUM, and KITTI Odometry runs to 4661 frames. We will fold these into a quantitative Limitations section.
