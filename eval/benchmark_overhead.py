"""
Benchmark inference overhead: wall-clock time + peak GPU memory.
Runs key configs on TUM (8 sequences) and reports per-config averages.

Usage:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python eval/benchmark_overhead.py \
        --weights model/cut3r_512_dpt_4_64.pth --size 512
"""

import os
import sys
import time
import json
import argparse
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from add_ckpt_path import add_path_to_dust3r
from dust3r.model import ARCroco3DStereo
from dust3r.inference import inference_recurrent_lighter
from dust3r.utils.image import load_images_for_eval as load_images
from eval.relpose.metadata import dataset_metadata


def prepare_input(img_paths, img_mask, size, revisit=1, update=True, crop=True):
    """Prepare views for inference (extracted from eval/relpose/launch.py)."""
    images = load_images(img_paths, size=size, crop=crop, verbose=False)
    views = []
    num_views = len(images)
    for i in range(num_views):
        view = {
            "img": images[i]["img"],
            "ray_map": torch.full(
                (images[i]["img"].shape[0], 6, images[i]["img"].shape[-2], images[i]["img"].shape[-1]),
                torch.nan,
            ),
            "true_shape": torch.from_numpy(images[i]["true_shape"]),
            "idx": i,
            "instance": str(i),
            "camera_pose": torch.from_numpy(np.eye(4).astype(np.float32)).unsqueeze(0),
            "img_mask": torch.tensor(True).unsqueeze(0),
            "ray_mask": torch.tensor(False).unsqueeze(0),
            "update": torch.tensor(True).unsqueeze(0),
            "reset": torch.tensor(False).unsqueeze(0),
        }
        views.append(view)
    return views


CONFIGS = [
    {"name": "cut3r", "update_type": "cut3r"},
    {"name": "ttt3r", "update_type": "ttt3r"},
    {"name": "ttt3r_random_p033", "update_type": "ttt3r_random", "random_gate_p": 0.33},
    {"name": "ttt3r_momentum_inv_t1", "update_type": "ttt3r_momentum", "momentum_tau": 1.0},
    {"name": "ttt3r_ortho", "update_type": "ttt3r_ortho"},
    {"name": "ttt3r_ortho_adaptive_linear", "update_type": "ttt3r_ortho", "ortho_adaptive": "linear"},
]


def set_config(model, cfg):
    """Set model config for a given benchmark config."""
    model.config.model_update_type = cfg["update_type"]
    model.config.random_gate_p = cfg.get("random_gate_p", 0.5)
    model.config.momentum_tau = cfg.get("momentum_tau", 2.0)
    model.config.ortho_alpha_novel = cfg.get("ortho_alpha_novel", 0.5)
    model.config.ortho_alpha_drift = cfg.get("ortho_alpha_drift", 0.05)
    model.config.ortho_beta = cfg.get("ortho_beta", 0.95)
    model.config.ortho_adaptive = cfg.get("ortho_adaptive", "")
    # Other defaults
    model.config.spectral_temperature = 1.0
    model.config.geo_gate_tau = 2.0
    model.config.geo_gate_freq_cutoff = 4
    model.config.gate_base_rate = 0.33
    model.config.gate_tau_sharp = 5.0
    model.config.novelty_tau = 5.0
    model.config.momentum_beta = 0.9
    model.config.momentum_lr = 0.33
    model.config.clip_alpha = 0.33
    model.config.clip_tau = 2.0
    model.config.clip_beta = 0.99
    model.config.attn_protect_beta = 0.95
    model.config.attn_protect_base = 0.33
    model.config.mem_novelty_base = 0.33
    model.config.mem_novelty_tau = 5.0
    model.config.mem_novelty_beta = 0.95


def benchmark_config(model, cfg, views_list, device, warmup=1, repeats=3):
    """Run benchmark for a single config. Returns timing and memory stats."""
    set_config(model, cfg)

    all_fps = []
    all_times = []
    all_peak_mem = []

    for rep in range(warmup + repeats):
        seq_fps = []
        seq_times = []
        seq_peak_mem = []

        for seq_name, views, n_frames in views_list:
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)

            start = time.perf_counter()
            with torch.no_grad():
                outputs, _ = inference_recurrent_lighter(views, model, device, verbose=False)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start

            peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 3)  # GB

            fps = n_frames / elapsed
            seq_fps.append(fps)
            seq_times.append(elapsed)
            seq_peak_mem.append(peak_mem)

        if rep >= warmup:
            all_fps.append(np.mean(seq_fps))
            all_times.append(np.sum(seq_times))
            all_peak_mem.append(np.max(seq_peak_mem))

    return {
        "fps_mean": float(np.mean(all_fps)),
        "fps_std": float(np.std(all_fps)),
        "total_time_mean": float(np.mean(all_times)),
        "total_time_std": float(np.std(all_times)),
        "peak_mem_gb": float(np.mean(all_peak_mem)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="model/cut3r_512_dpt_4_64.pth")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--eval_dataset", type=str, default="tum_s1_1000")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max_frames", type=int, default=200, help="Max frames per sequence for benchmark")
    parser.add_argument("--max_seqs", type=int, default=3, help="Max sequences to benchmark")
    parser.add_argument("--output", type=str, default="eval_results/benchmark_overhead.json")
    args = parser.parse_args()

    device = torch.device("cuda")

    # Load model
    print("Loading model...")
    model = ARCroco3DStereo.from_pretrained(args.weights)
    model.to(device)
    model.eval()

    # Prepare views for all sequences (load once, reuse)
    metadata = dataset_metadata.get(args.eval_dataset)
    img_path = metadata["img_path"]
    seq_list = metadata.get("seq_list", None)
    if seq_list is None or metadata.get("full_seq", False):
        seq_list = sorted([
            s for s in os.listdir(img_path)
            if os.path.isdir(os.path.join(img_path, s))
        ])
    else:
        seq_list = sorted(seq_list)

    seq_list = seq_list[:args.max_seqs]
    print(f"Preparing {len(seq_list)} sequences from {args.eval_dataset} (max {args.max_frames} frames)...")
    views_list = []
    for seq in seq_list:
        dir_path = metadata["dir_path_func"](img_path, seq)
        filelist = sorted([
            os.path.join(dir_path, name) for name in os.listdir(dir_path)
        ])[:args.max_frames]
        views = prepare_input(
            filelist, [True for _ in filelist],
            size=args.size, crop=True, revisit=1, update=True,
        )
        views_list.append((seq, views, len(filelist)))

    total_frames = sum(n for _, _, n in views_list)
    print(f"Total: {total_frames} frames across {len(views_list)} sequences")

    # Run benchmarks
    results = {}
    for cfg in CONFIGS:
        name = cfg["name"]
        print(f"\nBenchmarking {name} (warmup={args.warmup}, repeats={args.repeats})...")
        stats = benchmark_config(model, cfg, views_list, device,
                                 warmup=args.warmup, repeats=args.repeats)
        results[name] = stats
        print(f"  FPS: {stats['fps_mean']:.2f} ± {stats['fps_std']:.2f}")
        print(f"  Total time: {stats['total_time_mean']:.1f}s ± {stats['total_time_std']:.1f}s")
        print(f"  Peak GPU mem: {stats['peak_mem_gb']:.2f} GB")

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Config':<30} {'FPS':>8} {'Time(s)':>10} {'Peak Mem(GB)':>14} {'Overhead':>10}")
    print("-" * 80)

    base_fps = results["cut3r"]["fps_mean"]
    base_time = results["cut3r"]["total_time_mean"]

    for name, stats in results.items():
        overhead = (stats["total_time_mean"] / base_time - 1) * 100
        overhead_str = f"+{overhead:.1f}%" if overhead > 0 else f"{overhead:.1f}%"
        print(f"{name:<30} {stats['fps_mean']:>7.2f} {stats['total_time_mean']:>9.1f} "
              f"{stats['peak_mem_gb']:>13.2f} {overhead_str:>10}")

    print("=" * 80)

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
