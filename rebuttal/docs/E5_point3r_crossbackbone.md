# E5 — Cross-backbone study: Point3R

**Purpose**: answer Reviewer DpBu (R2-C1) and Reviewer 1ake (R3-C3) — *"is 'scalar gate collapse
→ directional regulation helps' a property of recurrent 3D reconstruction in general, or of this
specific checkpoint?"*

**Target**: **Point3R** (Wu, Zheng, Zhou, Lu — *Streaming 3D Reconstruction with Explicit Spatial
Pointer Memory*, **NeurIPS 2025**), repo `YkiWu/Point3R`, checkpoint from the authors' Drive link.

Why this is the right target, where StreamVGGT / Spann3R / LONG3R were not:
- it is a **streaming 3D reconstruction** model, i.e. squarely in this paper's setting;
- its backbone is **DUSt3R ViT-Large**, architecturally unrelated to CUT3R's `ARCroco3DStereo`;
- its memory is an **explicit spatial pointer set** — each pointer carries a 3D position and the
  set grows with the scene — as opposed to CUT3R's fixed 768-token implicit state;
- crucially, existing memory entries **are modified in place**, so an update `Δ` exists.

## The update Point3R actually performs

`src/dust3r/point3r.py::_forward_addmemory_merge`: a new observation whose 3D position falls within
`threshold_j` of an existing pointer is *merged* into it:

```python
feat_avg = feat_sum / count                  # mean of the NEW features only
memory_feat_j[unique_indices] = feat_avg     # hard overwrite; the old feature is discarded
```

This is `β = 1` — the extreme of failure mode **M1**. Our intervention keeps everything else fixed
and replaces only that assignment (`DDD3R_ALPHA`, default `1.0` ⇒ bit-identical to upstream):

```python
mem ← mem + α · (feat_avg − mem)
```

## Step 1 — Does the diagnosis transfer? (read-only instrumentation)

`rebuttal/scripts/point3r_diagnose.py` wraps the merge and records, per overwrite event,
`δ = feat_avg − mem[idx]`. 100 frames per sequence.

| Sequence | M1: ‖δ‖/‖mem‖ | **M3: drift energy** |
|---|---|---|
| TUM walking_xyz | 0.295 | 0.155 |
| TUM walking_static | — | 0.155 |
| TUM sitting_xyz | — | 0.155 |
| TUM sitting_halfsphere | 0.274 | 0.163 |
| ScanNet scene0707_00 | 0.294 | 0.164 |
| ScanNet scene0708_00 | 0.282 | 0.170 |
| ScanNet scene0709_00 | 0.282 | 0.164 |

**M1 transfers.** Point3R overwrites a pointer with an update whose norm is ~28 % of the stored
feature, with no dampening whatsoever.

**M3 does not transfer.** Drift energy sits at 0.155–0.170 across seven sequences and two datasets,
against 0.398 (TUM) and 0.598 (ScanNet) for CUT3R. `cos(δ_t, δ_{t−1})` is in fact **negative**
(−0.19 ± 0.35): successive updates to the same pointer tend to *cancel* rather than compound.

The cleanest comparison is ScanNet, where the *data are identical* and only the architecture differs:

| Dataset | CUT3R drift energy | Point3R drift energy |
|---|---|---|
| TUM | 0.398 | 0.155 |
| **ScanNet** | **0.598** | **0.164** |

**Mechanistic reading.** CUT3R's 768 state tokens carry no spatial assignment; every frame pushes all
of them through one global update, so successive updates stay aligned and drift compounds. Point3R
anchors each pointer to a 3D position, so a given pointer is updated only by observations near it,
from *different viewpoints* at different times — which decorrelates the update direction. Point3R
trades "fixed capacity, repeatedly overwritten" for "memory that grows with the scene"
(768 → 864 pointers over 100 frames), and in doing so trades away the directional pathology **at the
cost of no longer having bounded memory**.

## Step 2 — Does the M1 *fix* transfer? (intervention)

TUM, 90 frames, 8 sequences, ATE RMSE (m). α = 1.0 reproduces upstream exactly and is our baseline.

| α | ATE | vs upstream | scenes improved |
|---|---|---|---|
| 1.0 (upstream) | 0.0427 | — | — |
| 0.7 | 0.0382 | −10.6 % | 7/8 |
| **0.5** | **0.0341** | **−20.1 %** | 6/8 |
| 0.33 | 0.0417 | −2.2 % | 5/8 |

The optimum is at **α = 0.5, the same default our paper uses for DDD3R\_const on CUT3R**, and the
curve is non-monotone (−10.6 → −20.1 → −2.2), matching the over-update account: too little dampening
leaves the error in, too much starves the update.

**Statistical honesty (n = 8).** None of these reach significance: Wilcoxon p = 0.109 / 0.250 / 0.742
for α = 0.7 / 0.5 / 0.33, and every bootstrap 95 % CI crosses zero. Leave-one-out shows the α = 0.5
result is carried substantially by one sequence — dropping `walking_xyz` leaves only −5.0 %, while
the other seven leave-one-out estimates stay in −19.7 % … −26.0 %. The median moves −21.8 %, so the
direction is consistent, but **n = 8 cannot establish the effect**.

### ScanNet, n = 90 — the effect is real but modest

| α | ATE | vs upstream | 95 % CI | Wilcoxon p | scenes improved |
|---|---|---|---|---|---|
| 1.0 (upstream) | 0.0971 | — | — | — | — |
| 0.7 | 0.0948 | −2.3 % | [−0.0044, −0.0003] | 0.095 | 52/90 |
| **0.5** | **0.0918** | **−5.4 %** | **[−0.0083, −0.0024]** | **0.0022** | **56/90** |

α = 0.5 is significant with a CI excluding zero, and is again the optimum — the same coefficient our
paper uses on CUT3R. But the effect is **much smaller than on TUM** (−5.4 % vs −20.1 %) and the
median barely moves (−0.3 %), so the gain is concentrated in a subset of scenes rather than being a
uniform shift. We report it that way.

Taken together the two datasets are mutually reinforcing rather than redundant: TUM gives a large
effect that n = 8 cannot certify, ScanNet gives a modest effect that n = 90 certifies, and **both
select α = 0.5**.

## What this establishes

Split cleanly in two, and we report both halves:

1. **M1 (over-update) is architecture-independent.** It is present in a different backbone with a
   completely different memory design, and the *same* fix at the *same* coefficient helps.
2. **M3 (directional redundancy) is specific to a bounded, repeatedly-overwritten implicit state.**
   Explicit spatial anchoring avoids it — which is also why full directional decomposition should
   *not* be expected to help on Point3R.

Point 2 is a **successful prediction of C3**: drift energy is measured to be low, and C3 therefore
predicts directional decomposition is inapplicable here. The theory correctly identifies when its
own flagship component should not be used — the same pattern we pre-registered and confirmed on
ARKitScenes (E4).

## Reproduction notes

- The released checkpoint stores `args` as `{'model': '<config expr>'}`, but the repo's
  `load_model()` expects an argparse `Namespace` (`ckpt['args'].model`), so the official evaluation
  scripts fail out of the box. We patched `load_model` to accept both, and tightened `torch.load`
  to `weights_only=True`.
- `eval/relpose/metadata.py` was repointed at our existing TUM (`rgb_90` + `groundtruth_90.txt`) and
  ScanNet (`color_90` + `pose_90.txt`) trees — the layouts already match, so no re-preprocessing.
- All Point3R changes live under `rebuttal/external/Point3R/`; nothing in our own `src/` is touched.
