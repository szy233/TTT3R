# TTT3R — Frequency-Guided State & Memory Update Framework

## Project Goal
NeurIPS submission. Train-free, inference-time frequency-domain framework for selective state/memory updates in recurrent 3D reconstruction (CUT3R/TTT3R).

## Architecture Overview

The model (`src/dust3r/model.py`, class `ARCroco3DStereo`) processes video frames recurrently:
1. Encode frame → `feat_i`
2. `_recurrent_rollout(state_feat, feat_i)` → `new_state_feat`, `dec`
3. `pose_retriever.update_mem(mem, feat, pose)` → `new_mem`
4. `_downstream_head(dec)` → `res` (pts3d, conf)
5. State update: `state_feat = new * mask1 + old * (1-mask1)`
6. Memory update: `mem = new_mem * mask2 + mem * (1-mask2)`

`mask1` and `mask2` are where our frequency-domain gates are applied.

## Three-Layer Frequency Framework

### Layer 1 — Frame Filtering (validated)
- **Signal**: `LFE(FFT2(RGB_diff))` — low-freq energy of inter-frame RGB difference
- **Action**: Skip frames where LFE < threshold × EMA(LFE)
- **Result**: Skip 35% frames, TTT3R depth -3.1% on ScanNet
- **Code**: `compute_frame_spectral_change()`, `filter_views_by_spectral_change()`

### Layer 2 — Token-Level State Modulation (SIASU, validated)
- **Signal**: Per-token high-freq residual energy of state trajectory (EMA low-pass → residual)
- **Action**: `alpha_k = sigmoid(-τ × (energy_k / running_mean - 1))` per token
- **Code**: `_spectral_modulation()`, update types `cut3r_spectral` / `ttt3r_spectral`
- **Result**: cut3r_spectral -5.0%, ttt3r_spectral -8.3% (vs cut3r). τ insensitive, use τ=1
- **Status**: Validated (2026-03-23)

### Layer 3 — Geometric Consistency Gate (validated, best result)
- **Signal**: `LFE(FFT2(log_depth_diff))` — low-freq energy of log-depth change
- **Action**: Gate state update when depth prediction is geometrically inconsistent
- **Result**: cut3r_geogate -3.5%, ttt3r_geogate -7.2% (vs ttt3r -6.4%)
- **Code**: `_geo_consistency_gate()`, update types `cut3r_geogate` / `ttt3r_geogate`
- **Best config**: τ=2, freq_cutoff=4 (25% bandwidth). Cutoff-insensitive (c2/c4/c8 all ~-3.5%)

## Update Types in model.py

| `model_update_type` | `mask1` (state) | `mask2` (memory) |
|---------------------|-----------------|------------------|
| `cut3r` | 1.0 (baseline) | 1.0 |
| `ttt3r` | sigmoid(cross_attn) | 1.0 |
| `cut3r_spectral` | spectral_modulation α | 1.0 |
| `ttt3r_spectral` | ttt3r × α | 1.0 |
| `cut3r_memgate` | 1.0 | spectral_change gate |
| `ttt3r_memgate` | sigmoid(cross_attn) | spectral_change gate |
| `cut3r_geogate` | geo_consistency gate | 1.0 |
| `ttt3r_geogate` | ttt3r × geo gate | 1.0 |
| `cut3r_joint` | α × geo gate | 1.0 |
| `ttt3r_joint` | ttt3r × α × geo gate | 1.0 |

## Key Experimental Results

### B2 Memory Gate (weak, ~1% improvement)
```
cut3r_mg_t3_sr0.3   -1.75%   (best memgate variant)
ttt3r_mg_t3_sr0.5   -6.25%   (no gain over pure ttt3r -6.40%)
```
Memory's soft cross-attention write already handles redundancy. Direction deprioritized.

### B3 Geometric Consistency Gate (strong)
```
cut3r_geo_t2         -3.83%   (spatial domain, best)
cut3r_geo_t2_c4      -3.52%   (frequency domain)
ttt3r_geo_t3         -7.41%   (spatial domain, best overall)
ttt3r_geo_t2_c4      -7.16%   (frequency domain)
```

### Joint Ablation (L23+ttt3r is best)
```
cut3r (baseline)     0.0745   —
ttt3r (baseline)     0.0697   -6.4%
L1+ttt3r             0.0700   -6.0%
L2+ttt3r             0.0684   -8.2%
L3+ttt3r             0.0692   -7.2%
L23+ttt3r            0.0690   -7.5%   ← best combination
L123+ttt3r           0.0699   -6.2%   (L1 conflicts with L2/L3)
L23+cut3r            0.0698   -6.3%   (matches pure ttt3r)
```
L1 frame skipping conflicts with fine-grained L2/L3 modulation. Final method: L23+ttt3r.

### Failed Directions
- **Direction C (dynamic token tracking)**: State tokens don't track spatial semantics. Walking r=-0.024, static r=-0.383 (reversed). Abandoned.
- **Confidence gating (Exp 2)**: <1% improvement, feedback loop. Abandoned.

## Experiment Configs

All experiments share: `--seed 42 --size 512 --max_frames 200 --num_scannet 10`

Server paths:
- Model: `model/cut3r_512_dpt_4_64.pth`
- ScanNet: `/home/szy/research/dataset/scannetv2`
- TUM: `/home/szy/research/dataset/tum`
- Working dir: `/home/szy/research/TTT3R`

Local paths:
- Working dir: `/Users/shaozhengyu/code/TTT3R`
- Results synced to `analysis_results/` (gitignored)

Sync command: `rsync -avz 10.160.4.14:/home/szy/research/TTT3R/analysis_results/<exp>/ analysis_results/<exp>/`

## Key Files

| File | Purpose |
|------|---------|
| `src/dust3r/model.py` | All update types, gate methods, LocalMemory |
| `analysis/geogate_ablation.py` | B3 geo gate ablation |
| `analysis/memgate_ablation.py` | B2 memory gate ablation |
| `analysis/spectral_ablation.py` | Layer 2 SIASU ablation |
| `analysis/batch_frame_novelty.py` | Layer 1 validation |
| `analysis/metric_comparison.py` | spectral_change vs L2/high/mid freq |
| `analysis/joint_ablation.py` | Three-layer joint ablation |
| `docs/research_progress.md` | Full research log (Chinese) |
| `docs/run_experiments.sh` | All experiment commands |

## Known Issues / Fixes Applied
1. **SIASU warm-start**: `running_energy` init 0 → ratio explosion → state frozen. Fixed: warm-start on first call.
2. **TUM depth matching**: Timestamp-based association needed (not stem-based).
3. **Fair evaluation**: Compare full vs filtered on same `kept_indices`.

## Next Steps
1. ~~Re-run Layer 2 SIASU ablation (warm-start fixed)~~ Done (2026-03-23)
2. ~~Three-layer joint experiment (Layer 1 + 2 + 3)~~ Done (2026-03-23). L23+ttt3r -7.5% best; L1 conflicts.
3. Paper outline drafting
