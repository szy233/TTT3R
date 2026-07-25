# E4 — ARKitScenes: PRE-REGISTERED PREDICTION

> **This file is committed BEFORE any ATE on ARKitScenes is computed.**
> Its whole value is the timestamp. Check `git log` for this file against the
> creation time of `rebuttal/results/relpose/arkit_*/`. Nothing below may be
> edited after results exist — corrections go in a separate section at the end.

## Why this experiment

Reviewer ULz9: *"conclusions may not generalize well to real-world scenarios"* /
*"How does DDD3R perform on in-the-wild datasets?"*

The weak answer is "we added one more dataset and won again." The strong answer is to
show the paper's **theory** transfers: use C3 (drift energy governs the operating point)
to predict the ranking on unseen real-world data *before* measuring it.

## Dataset

**ARKitScenes** (Apple, 2021) — 5,048 RGB-D scans of 1,661 real indoor venues captured
handheld on a **2020 iPad Pro**. Ground-truth 6-DoF poses come from registering
**Faro Focus S70 laser scans**, so GT is independent of any visual SfM pipeline.

Selection, fixed in advance and free of cherry-picking:
- split = **Validation** (550 scenes after removing the 6 ids with missing assets)
- sort video_ids ascending as strings, take the **first 20**
- ids: 41069021 41069025 41069042 41069043 41069046 41069048 41069050 41069051
  41125696 41125700 41125709 41125718 41125722 41125731 41125756 41125760 41125763
  41142278 41142280 41142281
- evaluate the **first 1000 associated frames** of each scene
  → **amended to 500 frames before any ATE was computed; see "Amendment 1" below**

Preprocessing (`rebuttal/scripts/prepare_arkitscenes.py`), with the two failure modes
that would silently corrupt ATE handled explicitly:
- `.traj` stores **world→camera**; TUM `groundtruth.txt` wants **camera→world**, so we
  invert (verified against the official `TrajStringToMatrix`, which does the same).
- images run at ~60 Hz but poses only at ~10 Hz, so we drive from the **pose** stream and
  attach the nearest image within 50 ms — every evaluated frame has a *measured* pose,
  never an interpolated one.

Validation already run: our Rodrigues matches `cv2.Rodrigues` to 1.2e-15, quaternion
round-trip to 1.1e-15, and scene 41069021 gives a physically sensible trajectory
(1878 poses, 43.33 m path, 10.2 cm max step, 3.05×3.59×1.29 m extent, no NaN).

## The prediction

ARKitScenes is **slow handheld indoor scanning** — the same capture regime as ScanNet,
not the faster, more translational motion of TUM. Under C3, drift energy
`ē = mean cos²(δ_t, d_t)` should therefore be **high** (ScanNet-like, ē ≈ 0.6) rather
than moderate (TUM-like, ē ≈ 0.40).

**Primary prediction (P1).** Measured drift energy on ARKitScenes will satisfy
**ē > 0.50**, i.e. closer to ScanNet (0.598) than to TUM (0.398).

**Conditional prediction (P2)** — the real test, a *conditional* the paper's theory makes:

| if measured ē | then predicted ranking |
|---|---|
| **ē > 0.50** (ScanNet-like) | `brake < const < ortho` — isotropic dampening wins, full directional decomposition **loses**, mirroring ScanNet 1000f (0.261 / 0.283 / 0.488) |
| ē < 0.45 (TUM-like) | `ortho < brake < const` — directional decomposition wins, mirroring TUM 1000f (0.055 / 0.063 / 0.079) |

(`<` = lower ATE = better.)

**Prediction (P3).** Every DDD3R operating point beats **both** CUT3R and TTT3R, since M1
(over-update) is claimed to be regime-independent and has held on all six datasets so far.

## What each outcome means

- **P1 + P2 both hold** → C3 is *predictive on held-out real-world data*, not merely
  descriptive of the datasets we tuned on. This is the strongest possible answer to
  "do your conclusions generalize", and it is worth more than a win.
- **P2 holds but P1 fails** (ē turns out TUM-like *and* ortho wins) → the conditional
  still holds; only our guess about ARKitScenes' regime was wrong. C3 survives intact.
- **P2 fails** (ranking does not follow ē) → C3 does **not** transfer to unseen data.
  We report this. It bounds the claim, and we would soften C3 from "governs" to
  "governs within the studied benchmarks" in the camera-ready.
- **P3 fails** → M1 is not regime-independent. This would be the most serious outcome
  and we would report it as a limitation.

**We commit to reporting the outcome whichever way it goes.** The design deliberately
makes the honest result publishable: if the conditional holds, C3 is validated; if it
fails, we have found a real boundary of the claim, which is itself a finding and is
consistent with the paper already being a diagnosis-first contribution.

## Protocol (fixed now)

1. Measure drift energy on all 20 scenes with the existing analysis path. Record ē.
2. Write the resulting P2 branch into this file **before** running any pose eval.
3. Run `cut3r, ttt3r, ddd3r_constant, ddd3r_brake, ddd3r` on `arkit_s1_1000`.
4. Report ATE, ranking, and whether P1/P2/P3 held.

---

## Amendment 1 — sequence length 1000 → 500

**Made before any ATE on ARKitScenes was computed. No result influenced this change.**

The original "first 1000 frames" was extrapolated from the single scene downloaded at
registration time (41069021, 1878 poses). Having now converted all 20 scenes, the pose
streams are shorter than that scene suggested (median ≈ 900):

| length | scenes qualifying |
|---|---|
| 300 | 20/20 |
| **500** | **18/20** |
| 800 | 12/20 |
| 1000 | 8/20 |

Keeping 1000 would have cut the sample to n=8. We therefore evaluate at **500 frames on
the 18 qualifying scenes** (all except 41069048 with 330 and 41069050 with 311).

Why this does not weaken the pre-registration:
- The change is driven purely by **data availability**, which is independent of any
  method's performance — no ATE existed when this was written.
- The scene *selection rule* (Validation split, video_id ascending, first 20) is unchanged.
- 500 frames remains squarely in the long-sequence regime this paper studies: the main
  video-depth tables (KITTI, Bonn) are themselves reported at 500 frames, and on TUM /
  ScanNet the methods are already well separated by 500 frames.
- P1, P2 and P3 are **unchanged** — the predictions are about drift energy and ranking,
  not about a particular length.

Excluded scenes (too short for 500f): 41069048 (330), 41069050 (311).
Final evaluation set: **18 scenes × 500 frames, 640×480 (`vga_wide`)**.

## Data preparation outcome (pre-ATE, factual record)

All 20 scenes converted with `--img-asset vga_wide`: **100% of pose timestamps
associated to an image within 50 ms** in every scene (e.g. 1878/1878, 1675/1675,
966/966, ...). Resolution is 640×480, matching TUM, so no low-resolution caveat applies.

---

## P2 BRANCH LOCKED — drift energy measured, ranking predicted, NO ATE YET

**Measured before any ARKitScenes ATE existed.** `rebuttal/results/arkit_drift_energy.json`,
produced by `rebuttal/scripts/arkit_drift_energy.py`, which calls the paper's own
`analysis/a4_delta_direction.py:run_detailed_delta_analysis` so the number is directly
comparable to Table 3.

```
ARKitScenes drift energy: 0.632 ± 0.035   (n = 18 scenes, 500 f, vga_wide 640×480)
                     cos: 0.790 ± 0.023
reference (Table 3):  TUM 0.398 ± 0.041 | ScanNet 0.598 ± 0.054
```

### P1 — HOLDS
Predicted `ē > 0.50`; measured **0.632**. ARKitScenes is not merely ScanNet-like, it is
*more* drift-dominated than ScanNet (0.598), and far from TUM (0.398). Consistent with its
capture protocol: slow handheld indoor scanning with substantial revisiting.
The per-scene spread is tight (σ = 0.035, all 18 scenes in 0.55–0.66), so the regime
assignment is unambiguous rather than an artifact of averaging.

### P2 — ScanNet-like branch selected. **Predicted ranking, locked now:**

> **DDD3R_brake < DDD3R_const < DDD3R_ortho**   (`<` = lower ATE = better)

That is: **isotropic dampening should win, and full directional decomposition should
*lose***, mirroring ScanNet 1000f (brake 0.261 < const 0.283 < ortho 0.488).

This is a deliberately **falsifiable and counter-intuitive** prediction: it forecasts that our
own flagship variant (ortho) will be the *worst* of the three DDD3R operating points on this
dataset. Any outcome in which ortho beats brake falsifies P2.

Secondary expectation (not part of the pass/fail criterion): given ē = 0.632 exceeds
ScanNet's 0.598, the ortho penalty should be at least as pronounced as on ScanNet.

### P3 — unchanged
All DDD3R operating points beat both CUT3R and TTT3R.

**Nothing below this line may be written until the ATE runs have completed.**

---

## Post-hoc section (to be appended only after results exist)

*(empty)*
