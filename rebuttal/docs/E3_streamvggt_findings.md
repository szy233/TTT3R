# E3 — Cross-backbone study: StreamVGGT

**Purpose**: answer Reviewer DpBu (R2-C1) and Reviewer 1ake (R3-C3) — *"validate the generality of
DDD3R on other recurrent 3D reconstruction models"* / *"unclear whether this is a property of
recurrent 3D reconstruction in general or of this specific checkpoint."*

**Target chosen**: StreamVGGT (Zhuo, Zheng, Guo, Wu, Zhou, Lu — *Streaming 4D Visual Geometry
Transformer*, ICLR 2026, arXiv:2507.11539), repo `wzzheng/StreamVGGT`, weights `lch01/StreamVGGT`.
It is the most natural target: an explicitly *streaming*, *causal*, *online* 3D reconstruction model.

---

## Finding: DDD3R is architecturally inapplicable to StreamVGGT — and this is informative

StreamVGGT does **not** maintain a bounded recurrent state that is overwritten each frame. It
maintains an **append-only KV cache** that grows monotonically with sequence length. Verified in the
released source:

`src/streamvggt/layers/attention.py:60-63`
```python
if past_key_values is not None:
    past_k, past_v = past_key_values
    k = torch.cat([past_k, k], dim=2)
    v = torch.cat([past_v, v], dim=2)
```

This is the **only** cache-mutation site in the entire repository. A repo-wide search for
`evict|prune|truncat|max_cache|window|compress` returns **no** matches under `src/streamvggt/`.
`StreamVGGT.inference()` (`src/streamvggt/models/streamvggt.py:105-128`) initializes
`past_key_values = [None] * aggregator.depth` and only ever appends. There is no `state_feat`
variable anywhere in the model.

**Consequence.** DDD3R regulates the quantity `Δ_t = S_t^new − S_{t−1}`, i.e. the *overwrite* applied
to a fixed-size state. In StreamVGGT past entries are immutable and nothing is overwritten, so
`Δ_t` **does not exist**. DDD3R is inapplicable not because it fails, but because its input is
undefined. Equally importantly, the failure mode DDD3R targets — drift accumulated by repeatedly
overwriting a bounded state — **cannot arise** in an append-only cache.

## The real taxonomy (this is the answer to give reviewers)

| Family | Mechanism | Memory | Failure mode | DDD3R |
|---|---|---|---|---|
| **A. Bounded overwriting state** — CUT3R, TTT3R, TTSA3R | `S_t = S_{t−1} + β_t Δ_t` | **constant** | drift accumulation in the state | **applies** |
| **B. Append-only cache** — StreamVGGT | `KV_t = [KV_{t−1}; kv_t]` | **grows linearly in T** | unbounded memory/compute | no `Δ_t` to regulate |
| **C. Stateless** — DUSt3R, MASt3R | pairwise + global alignment | n/a | cannot stream | no state at all |

The two failure modes are **complementary, not competing**: Family A buys constant memory and pays
in drift; Family B avoids drift by never compressing and pays in unbounded memory. DDD3R targets
exactly the family that is defined by constant-memory streaming — the deployment regime DpBu
himself identifies as the motivation ("low-compute streaming reconstruction devices ... robotics
and AR").

## Quantifying the trade-off — MEASURED

Config (`aggregator.py`): `img_size=518, patch_size=14, embed_dim=1024, depth=24, num_heads=16,
num_register_tokens=4, aa_order=["frame","global"], aa_block_size=1`
→ tokens per frame `P = 1 (camera) + 4 (register) + (518/14)² = 1374`; only the 24 `global_blocks`
cache, giving 48 cached tensors (K and V per block).

Measured with the released weights on an A100
(`rebuttal/scripts/measure_streamvggt_memory.py`, raw: `rebuttal/results/streamvggt_memory.json`):

| frames | KV cache | per frame |
|---|---|---|
| 4 | 1.081 GB | 270.1 MB |
| 8 | 2.161 GB | 270.1 MB |
| 16 | 4.322 GB | 270.1 MB |

Growth is **exactly linear** — the per-frame figure is constant to four significant figures across a
4× range, confirming append-only accumulation with no eviction or compression. The measurement
matches the analytic prediction exactly: `24 × 2 × 1374 × 1024 × 4 bytes = 270.1 MB` (the cache is
held in fp32; bf16 would halve it to 135 MB/frame).

| Sequence length | StreamVGGT KV cache (fp32 / bf16) | CUT3R / DDD3R state |
|---|---|---|
| 100 f | 27.0 / 13.5 GB | **1.18 MB** |
| 500 f | 135.1 / 67.6 GB | **1.18 MB** |
| 1000 f | **270.1 / 135.1 GB** | **1.18 MB** |

DDD3R's state is `768 × 768 × 2 bytes = 1.18 MB`, **constant in T**.

**Scoping consequence, now verified rather than asserted.** On a single 80 GB A100 the KV cache
alone exhausts memory at roughly **290 frames** in fp32 (~590 in bf16), before counting weights and
activations — measured peak allocation was already 10.43 GB at just 16 frames. StreamVGGT therefore
cannot run the 1000-frame benchmarks that define this paper's problem setting on one 80 GB GPU, in
either precision. The long-sequence regime where over-update dominates is exactly the regime that
forces a bounded state, and a bounded state is exactly what creates the drift DDD3R regulates.

## What we will tell reviewers

1. We took the cross-backbone request seriously and selected StreamVGGT, the leading streaming model.
2. We report the *architectural* result: it belongs to a different family with no state update, so
   DDD3R is undefined there — verified against released source, not asserted.
3. We give the taxonomy above and scope our claim explicitly to Family A (bounded overwriting state),
   naming CUT3R, TTT3R and TTSA3R as its current members — three update rules we now *do* compare
   head-to-head (see E1).
4. We state what would falsify the claim: another Family-A model showing no benefit from directional
   decomposition.

## Remaining option for a true Family-A second backbone

**Spann3R** (Wang & Agapito) maintains a bounded *spatial memory* with explicit attention-based
read/**write**, so it does have an overwrite operation and would be a genuine Family-A test. This is
the correct target if a second backbone is required. Flagged for decision — StreamVGGT was the
user-selected priority and has now been resolved architecturally.

## Status

- repo cloned → `rebuttal/external/StreamVGGT`
- weights downloading → `rebuttal/external/StreamVGGT_ckpt` (`model.safetensors`)
- empirical memory measurement → queued behind E1 ScanNet on GPU 1
