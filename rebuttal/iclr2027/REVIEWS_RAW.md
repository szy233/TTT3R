# Raw reviews — ICLR 2027 Submission 28147 (DDD3R)

Verbatim as pasted by the author on 2026-11-06. Inline math was lost when the text was copied from OpenReview, which is why some sentences have gaps such as "using . For ,". Read the original page for those symbols.

---

Official Review of Submission28147 by Reviewer qpVA
Official Reviewby Reviewer qpVA22 Oct 2026 at 13:25 (modified: 06 Nov 2026 at 20:26)Everyone[Revisions](https://openreview.net/revisions?id=iVSocupnnc)
Paper Summary:
This paper proposes DDD3R, a training-free state-update regulator for recurrent 3D reconstruction. It decomposes each token’s update into components parallel and orthogonal to an EMA-tracked update direction, and applies stronger attenuation to the parallel component. The paper also studies constant dampening and a scalar temporal-brake variant. Experiments cover camera pose estimation, video depth prediction, and 3D reconstruction, with additional evaluations on ARKitScenes and Point3R.
Presentation: 2: Generally understandable, but requires substantial effort in places.
Informativeness: 2: I learned something interesting or useful.
Soundness: 2: I have minor or fixable concerns about the paper’s soundness.
Critical Strengths:

1. The results are promising, and the analysis is thorough. The work aligns well with the scope of the conference.
2. Constant dampening provides an informative baseline. The sequence-length experiments show how reconstruction errors accumulate over time.
3. Experiments cover pose estimation, depth prediction, and 3D reconstruction. ARKitScenes further tests the proposed diagnosis on unseen data.

Critical Weaknesses:

1. The framework largely shifts the problem to hyperparameter search. The preferred configuration changes across datasets, yet no reliable rule is provided for selecting one on unseen scenes.
2. The directional variant does not consistently outperform existing methods. On ScanNet-1000, its ATE is 0.488, worse than both TTT3R (0.406) and TTSA3R (0.378), and it also underperforms TTT3R on ARKitScenes.
3. The added value of directional decomposition remains unclear. Constant dampening already achieves substantial gains, while the most robust variant still uses scalar gating, leaving the need for directional control insufficiently demonstrated.
4. The gate-collapse diagnosis overstates the limitations of prior methods. Stable frame-averaged gates do not rule out useful token-level temporal adaptation in TTT3R or TTSA3R.
5. The mathematical description is internally inconsistent. The EMA timing differs between the equations and Figure 2, while the claimed γ interpolation direction contradicts Equation (12).

Questions For Authors:
see weaknesses.
Flag For Ethics Review: No
Rating: 2: Weak rejection
Confidence: 3: I am familiar with this topic and its relevant literature.
Official Review of Submission28147 by Reviewer PSN7
Official Reviewby Reviewer PSN722 Oct 2026 at 09:49 (modified: 06 Nov 2026 at 20:26)Everyone[Revisions](https://openreview.net/revisions?id=9apN3ncFCN)
Paper Summary:
The paper proposes that the degradation of 3D estimation on long sequences is due to three factors: (1) latent state is overly updated, (2) the gate change along the time is small, (3) update direction is repeated. Then it proposes to decompose the update of each token into the parallel and orthogonal directions w.r.t the history direction.
Presentation: 3: Clear and easy to follow.
Informativeness: 3: I gained a substantially deeper understanding of the problem or question.
Soundness: 3: I have no substantive concerns about the paper’s soundness.
Critical Strengths:

1. The paper provides valuable insights and visions on how the updates of latent states affect the streaming 3D estimators. It found that reducing the update brings improvement and further studied the cause behind. A carefully designed direction decomposition is proposed to improve the state update.
2. The method is training-free and plug-and-play on related streaming base models, which is practical and useful. 
3. The degradation on ScanNet, Sintel, and ARKitScenes are also reported, demonstrating the limitation and point out the future improvement direction.

Critical Weaknesses:
The improvement from direction decomposition is not cleanly isoluated or ablated. The brake baseline is quite good without direction decomposition. Hence we couldn't clearly tell the improvement is from less pace or corrected direction.
Questions For Authors:
If the per-token update follows the same norm as ortho without direction decomposition, how is the performance on TUM and ScanNet? Would that change the conclusion?
Flag For Ethics Review: No
Rating: 3: Weak acceptance
Confidence: 2: I have broad familiarity with this field but little or no direct research experience on this topic.
Official Review of Submission28147 by Reviewer MDAW
Official Reviewby Reviewer MDAW16 Oct 2026 at 15:25 (modified: 06 Nov 2026 at 20:26)Everyone[Revisions](https://openreview.net/revisions?id=pqaZZKZ1vD)
Paper Summary:
This paper investigates the state-update dynamics of recurrent 3D reconstruction models, focusing on long-sequence performance degradation. The authors identify three potential issues: excessive update magnitudes, limited temporal variation in existing adaptive gates, and directional redundancy among consecutive state updates.
To address these issues, the paper proposes DDD3R, a training-free framework that decomposes per-token state updates into components parallel and orthogonal to an exponentially averaged update direction. The two components are then reweighted separately to suppress redundant updates while preserving potentially novel information.
The authors investigate several configurations, including constant dampening, temporal braking, and explicit directional decomposition. Experiments on TUM, ScanNet, KITTI, Bonn, Sintel, and 7Scenes demonstrate improvements in camera pose estimation, video depth prediction, and 3D reconstruction. Additional analyses examine the relationship between drift energy and the effectiveness of different update strategies, including evaluations on ARKitScenes and Point3R.
The central contribution is an empirical analysis of recurrent state-update dynamics, together with a lightweight inference-time regulation strategy intended to improve long-sequence reconstruction without retraining.
Presentation: 2: Generally understandable, but requires substantial effort in places.
Informativeness: 2: I learned something interesting or useful.
Soundness: 2: I have minor or fixable concerns about the paper’s soundness.
Critical Strengths:
S1. A practically valuable diagnosis of recurrent state-update dynamics.
The paper provides useful empirical evidence that excessive state updates contribute substantially to long-sequence reconstruction degradation. In particular, the observation that simple constant dampening already produces substantial improvements is informative and highlights an important issue in recurrent reconstruction architectures. The analysis of update magnitudes, gate statistics, and directional similarity provides a useful perspective for understanding long-horizon state accumulation.
S2. Strong practical improvements without retraining.
The proposed regulation strategies are lightweight and require no additional learnable parameters. DDD3Rbrake reduces ATE from 0.103 to 0.063 on TUM and from 0.406 to 0.261 on ScanNet relative to TTT3R at 1000 frames. The improvements across multiple reconstruction-related tasks suggest practical value. The comprehensive evaluation across multiple datasets and tasks provides valuable insights into the effectiveness and limitations of different update regulation strategies.
Critical Weaknesses:
W1. The empirical results do not sufficiently establish the independent contribution of directional decomposition.
The central methodological contribution is the separation of state updates into drift-aligned and orthogonal components. However, the strongest and most consistent results are often obtained by DDD3Rbrake, which applies scalar temporal modulation without explicit directional decomposition.
The full directional variant improves TUM 1000-frame ATE from 0.063 to 0.055 relative to DDD3Rbrake, but performs substantially worse on ScanNet (0.488 versus 0.261) and also degrades performance on ARKitScenes relative to simpler alternatives.
Although the authors acknowledge these dataset-dependent effects, the evidence does not yet establish when directional decomposition provides a reliable advantage over simpler scalar regulation. This distinction is particularly important because the default directional hyperparameters were selected using TUM, which contains only eight evaluation sequences.
W2. Several mathematical inconsistencies weaken the claimed unified formulation.
Equation (12) defines the adaptive parallel coefficient using . For , increasing  causes this term to approach zero, so the resulting coefficient approaches , corresponding to stronger directional differentiation rather than isotropic dampening. This contradicts the interpretation of the parameter sweep in Appendix A.4. Moreover, the special treatment of  in Algorithm 1 introduces a discontinuity in the claimed continuous spectrum.
Equation (7) computes the EMA drift direction using the current update before performing the projection. Consequently, setting  makes the reference direction equal to the current normalized delta, rather than the previous delta as assumed in Appendix A.8.
Questions For Authors:
Q1. What is the actual implementation of the drift-energy-adaptive interpolation?
Equation (12) implies that increasing  moves the method toward the directional configuration rather than isotropic dampening, contrary to the discussion in Appendix A.4. Could the authors clarify whether the experiments use the stated equation and explain the discontinuity between  and ?
Q2. Which EMA direction is used for projection, and how is temporal braking related to the unified formulation?
Algorithm 1 appears to construct  using the current delta before projection, whereas Appendix A.8 assumes that setting  recovers the previous update direction. Could the authors clarify the implementation and provide a comparison between projecting onto  and the updated ?
Additionally, is DDD3Rbrake intended to be an exact special case of Equation (6), or only a qualitatively related scalar baseline?
Q3. Can the authors provide stronger evidence for the independent benefit of directional decomposition?
Since DDD3Rbrake outperforms DDD3Rortho on several benchmarks, the distinction between scalar regulation and directional decomposition is central to the paper's contribution.
Could the authors report per-sequence paired comparisons, confidence intervals, and statistical significance for DDD3Rortho versus DDD3Rbrake on TUM? Given that the default directional parameters were tuned on TUM, additional evidence from independently selected hyperparameters or held-out sequences would also help establish whether the reported improvement generalizes.
Flag For Ethics Review: No
Rating: 2: Weak rejection
Confidence: 2: I have broad familiarity with this field but little or no direct research experience on this topic.
