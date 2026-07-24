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

## Post-hoc section (to be appended only after results exist)

*(empty)*
