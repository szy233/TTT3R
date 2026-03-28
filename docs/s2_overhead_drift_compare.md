# S2 Overhead Comparison: `drift>0` vs `drift0` (SAFE224, Local)

## 1. Objective and Hypothesis
**Objective.** Quantify whether keeping residual drift (`alpha_drift > 0`) introduces extra inference overhead versus fully removing drift (`alpha_drift=0`) under the same local SAFE224 protocol.

**Hypothesis.** If the brake design is lightweight, `drift>0` and `drift0` should have near-identical runtime/per-frame cost.

## 2. Experimental Conditions

### 2.1 Hardware / Software
- OS: WSL2 Ubuntu 22.04
- Python: 3.10.12
- PyTorch: 2.5.1+cu121 (CUDA 12.1)
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB
- Driver: 576.52

### 2.2 Data and Video Resolution
- Dataset slice: sampled single-object sequences (`apple`, `bottle`)
- Sequence lengths: 12 / 24 frames
- Number of unique sequences: 4
- Original frame resolutions:
  - `540_79043_153212_len012/024`: `2000×900`
  - `618_100690_201667_len012/024`: `1108×2000`
- **Model input resolution:** `224` (SAFE224 setting, images are resized before inference)

### 2.3 Model and Runtime Parameters
- Checkpoint: `model/cut3r_512_dpt_4_64.pth`
- Device: `cuda`
- `frame_interval=1`
- `reset_interval=100`
- `downsample_factor=100`
- `model_update_type=ttt3r_momentum_inv_t1`

### 2.4 Compared Methods
- `ttt3r_momentum_inv_t1` (`alpha_drift=0.15`, i.e., drift retained)
- `ttt3r_momentum_inv_t1_drift0` (`alpha_drift=0.0`, i.e., drift removed)

### 2.5 Repeat Design
- Fixed-seed protocol: `42, 42, 42`
- Different-seed protocol: `41, 42, 43`
- Total runs: `2 methods × 4 sequences × 6 repeats = 48`
- Execution status: all completed, `timed_out=0`

## 3. Metrics and Statistics
- Overhead metrics:
  - `runtime_sec`
  - `per_frame_sec`
- Quality snapshot metrics (for context):
  - `basic_consistency_score`
  - `loop_closure_trans_error`
- Reporting format: mean ± std over repeated runs.

## 4. Results

### 4.1 Overall Overhead (mean ± std)
| Protocol | Method | Runtime (s) | Per-frame (s) |
|---|---|---:|---:|
| fixed_seed | `ttt3r_momentum_inv_t1` | `15.6084 ± 1.7761` | `0.9508 ± 0.2819` |
| fixed_seed | `ttt3r_momentum_inv_t1_drift0` | `15.3243 ± 1.3091` | `0.9327 ± 0.2562` |
| different_seed | `ttt3r_momentum_inv_t1` | `14.9859 ± 1.4224` | `0.9086 ± 0.2386` |
| different_seed | `ttt3r_momentum_inv_t1_drift0` | `14.9558 ± 1.3446` | `0.9084 ± 0.2436` |

### 4.2 By Sequence Length (different_seed)
| Method | len=12 per-frame (s) | len=24 per-frame (s) |
|---|---:|---:|
| `ttt3r_momentum_inv_t1` | `1.1367 ± 0.0152` | `0.6805 ± 0.0110` |
| `ttt3r_momentum_inv_t1_drift0` | `1.1410 ± 0.0247` | `0.6758 ± 0.0096` |

### 4.3 Quality Snapshot (overall mean)
| Method | Basic Consistency | Loop Trans Error |
|---|---:|---:|
| `ttt3r_momentum_inv_t1` | `1.1085` | `0.3946` |
| `ttt3r_momentum_inv_t1_drift0` | `1.1083` | `0.3903` |

### 4.4 Effect Size (protocol-averaged)
Comparing `drift0` to `drift>0` (`alpha_drift=0.15`):
- Runtime: `-1.03%`
- Per-frame: `-0.98%`
- Basic consistency: `-0.018%`
- Loop trans error: `-1.08%`

These differences are all small in magnitude on this local subset.

## 5. Paper-Oriented Analysis
1. **Efficiency claim is supported.**  
   Keeping non-zero drift does not introduce measurable latency overhead under SAFE224 local settings.

2. **Runtime jitter exists but is method-agnostic.**  
   Variance is present in both methods/protocols and is consistent with normal local scheduling/GPU runtime noise, not method-specific compute inflation.

3. **Overhead and quality can be discussed separately.**  
   This experiment is intentionally an S2 efficiency check. It shows cost neutrality of `alpha_drift`, while final quality ranking should rely on larger multi-dataset evaluations.

## 6. Limitations
- This is a local, small-scale subset (4 sequences).  
- Peak VRAM was not reliably captured in this run (NVML monitor unavailable in current logger output), so this report focuses on wall-clock overhead.

## 7. Conclusion
Under controlled SAFE224 local repeats, `alpha_drift=0.15` and `alpha_drift=0.0` have **nearly identical inference overhead**.  
This supports the practicality of retaining drift in the brake design without extra runtime burden.
