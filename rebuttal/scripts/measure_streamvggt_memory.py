#!/usr/bin/env python3
"""
E3 — Empirically measure StreamVGGT's KV-cache growth vs. DDD3R's constant state.

Measures the ACTUAL bytes held in past_key_values after N frames, for several N,
then reports the per-frame slope. Extrapolation is linear and exact because the
cache is a plain concatenation (verified: attention.py:60-63 is the only mutation
site, and there is no eviction/compression anywhere in src/streamvggt/).

Small N is sufficient and keeps this runnable alongside other jobs.

Usage:
  PYTHONPATH=rebuttal/external/StreamVGGT/src \
  python rebuttal/scripts/measure_streamvggt_memory.py --frames 2 4 8 16
"""
import argparse
import json
import os
import sys

import torch


def cache_bytes(past_key_values):
    """Total bytes held across every cached K/V tensor."""
    total, n_tensors = 0, 0
    for entry in past_key_values or []:
        if entry is None:
            continue
        for t in entry:
            if torch.is_tensor(t):
                total += t.numel() * t.element_size()
                n_tensors += 1
    return total, n_tensors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, nargs="+", default=[2, 4, 8, 16])
    ap.add_argument("--ckpt", default="rebuttal/external/StreamVGGT_ckpt/model.safetensors")
    ap.add_argument("--res", type=int, default=518)
    ap.add_argument("--out", default="rebuttal/results/streamvggt_memory.json")
    args = ap.parse_args()

    from streamvggt.models.streamvggt import StreamVGGT

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32

    model = StreamVGGT()
    if os.path.exists(args.ckpt):
        from safetensors.torch import load_file
        sd = load_file(args.ckpt)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"loaded ckpt: {len(missing)} missing, {len(unexpected)} unexpected")
    else:
        print(f"WARNING: ckpt not found at {args.ckpt} — measuring an untrained model. "
              f"Cache SIZES are architecture-determined and unaffected, but say so in the writeup.")
    model = model.to(dev).eval()

    results = []
    for n in args.frames:
        torch.cuda.empty_cache() if dev == "cuda" else None
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        frames = torch.rand(1, n, 3, args.res, args.res, device=dev, dtype=dtype)
        with torch.no_grad():
            past_kv = [None] * model.aggregator.depth
            total_b = 0
            for i in range(n):
                out = model.aggregator(
                    frames[:, i:i + 1], past_key_values=past_kv, use_cache=True
                )
                past_kv = out[2]
            total_b, n_tensors = cache_bytes(past_kv)
        peak = torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else float("nan")
        rec = {
            "frames": n,
            "kv_cache_bytes": total_b,
            "kv_cache_GB": total_b / 1e9,
            "per_frame_MB": total_b / n / 1e6,
            "n_cached_tensors": n_tensors,
            "peak_alloc_GB": peak,
        }
        results.append(rec)
        print(f"N={n:>3}  cache={rec['kv_cache_GB']:.3f} GB  "
              f"per-frame={rec['per_frame_MB']:.1f} MB  "
              f"tensors={n_tensors}  peak={peak:.2f} GB")
        del frames
        if dev == "cuda":
            torch.cuda.empty_cache()

    if len(results) >= 2:
        slope = ((results[-1]["kv_cache_bytes"] - results[0]["kv_cache_bytes"])
                 / (results[-1]["frames"] - results[0]["frames"]))
        print(f"\nmeasured slope: {slope/1e6:.1f} MB/frame")
        for T in (100, 500, 1000):
            print(f"  extrapolated @ {T:>4}f: {slope*T/1e9:.1f} GB")
        print("\nDDD3R / CUT3R state: 768 x 768 x 2 bytes = "
              f"{768*768*2/1e6:.2f} MB, CONSTANT in T")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
