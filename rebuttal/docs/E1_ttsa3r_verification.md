# E1 — TTSA3R baseline: implementation provenance

**Purpose**: answer Reviewer 1ake (R3-C2) — *"TTSA3R / TAUM is discussed analytically (M2) but is
not given a head-to-head row ... leaving the prediction 'TAUM ≡ constant dampening' empirically
unconfirmed."*

## What we did

Added `model_update_type = "ttsa3r"` to `src/dust3r/model.py` (in `forward_recurrent_lighter`,
the dispatch the evaluation path actually uses), so TTSA3R can be **scored**, not just logged.

## Verification against the official release

Checked line-by-line against the official TTSA3R implementation at
`/home/szy/research/TTSA3R/src/dust3r/model.py:1142-1159`. Every operation matches:

| Step | Official (TTSA3R repo) | Ours |
|---|---|---|
| state change | `(new_state_feat - prev_new_state_feat).norm(dim=-1).squeeze(0)` | identical |
| normalize | `state_change / state_change.mean()` | identical |
| TAUM temporal gate | `sigmoid(state_change_normalized - 1.5)` | identical (`taum_tau` default 1.5) |
| feature dissimilarity | `1.0 - (feat_i_norm * prev_feat_norm).sum(dim=-1)` | identical |
| cross-attn | `rearrange(cat(...), 'l h nstate nimg -> 1 nstate nimg (l h)')[:, :, 1:, :]` | identical |
| attention magnitude | `cross_att.mean(dim=-1).abs()` | identical |
| SCUM spatial signal | `(attn_mag * feat_dissim.unsqueeze(1)).max(dim=-1)[0].squeeze(0)` | identical |
| SCUM spatial gate | `sigmoid(spatial_signal)` | identical |
| fusion | `temporal_mask * spatial_mask` → `[1, 768, 1]` | identical |

**One intentional difference**: the official code sets `update_mask1 = final_mask` directly,
whereas we use `update_mask * final_mask`, preserving this codebase's `img_mask`/`update` flags
(the same convention the `ttt3r` branch here uses). On all benchmarks in this paper every frame has
`img_mask = True`, so `update_mask = 1.0` and the two are numerically identical. This keeps the
baseline consistent with how every other method in our tables is gated.

**Note on the backbone question (R2-C1 / R3-C3)**: the official TTSA3R repo is itself built on the
**same** `ARCroco3DStereo` (CUT3R) backbone as TTT3R and this work. TTSA3R is a different *gate*, not
a different *backbone* — which supports the paper's framing of TTT3R/TTSA3R/DDD3R as competing update
rules over a shared recurrent architecture.

## Runs

| Dataset | Status |
|---|---|
| Sintel (relpose, 14 seqs, ~50f) | ✅ complete — smoke test + reportable |
| TUM 1000f (relpose, 8 seqs) | launched |
| ScanNet 1000f (relpose, 96 scenes) | queued |

Outputs under `rebuttal/results/relpose/<dataset>/ttsa3r/`, kept separate from the main
`eval_results/` tree.

## Prediction under test

The paper's M2 claims TAUM collapses to a near-constant ≈ `sigmoid(1.0 - 1.5) = sigmoid(-0.5) = 0.378`
with `σ_temporal ≈ 0.006`, i.e. that TTSA3R is *effectively constant dampening at an untuned operating
point*. The head-to-head prediction is therefore that TTSA3R should land **near DDD3R_const
(α = 0.5)** and **well behind** DDD3R_brake / DDD3R_ortho — not near TTT3R.

This is a falsifiable prediction registered before the numbers were read.
