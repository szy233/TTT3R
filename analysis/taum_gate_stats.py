"""
Analyze TTSA3R's TAUM gate statistics using TTT3R model with cut3r_taum_log mode.

Usage:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python analysis/taum_gate_stats.py --dataset tum
"""
import os, sys, argparse, json, torch, numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
from add_ckpt_path import add_path_to_dust3r
from dust3r.model import ARCroco3DStereo
from dust3r.inference import inference_recurrent_lighter
from dust3r.utils.image import load_images_for_eval as load_images


def prepare_input(img_paths, size, crop=True):
    images = load_images(img_paths, size=size, crop=crop, verbose=False)
    views = []
    for i in range(len(images)):
        view = {
            "img": images[i]["img"],
            "ray_map": torch.full((images[i]["img"].shape[0], 6, images[i]["img"].shape[-2], images[i]["img"].shape[-1]), torch.nan),
            "true_shape": torch.from_numpy(images[i]["true_shape"]),
            "idx": i, "instance": str(i),
            "camera_pose": torch.from_numpy(np.eye(4).astype(np.float32)).unsqueeze(0),
            "img_mask": torch.tensor(True).unsqueeze(0),
            "ray_mask": torch.tensor(False).unsqueeze(0),
            "update": torch.tensor(True).unsqueeze(0),
            "reset": torch.tensor(False).unsqueeze(0),
        }
        views.append(view)
    return views


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="tum")
    parser.add_argument("--weights", type=str, default="model/cut3r_512_dpt_4_64.pth")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--max_frames", type=int, default=90)
    parser.add_argument("--max_seqs", type=int, default=8)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()
    if not args.output:
        args.output = f"analysis_results/taum_gate_stats/{args.dataset}_stats.json"

    device = torch.device("cuda")
    print(f"Loading model from {args.weights}...")
    model = ARCroco3DStereo.from_pretrained(args.weights)
    model.to(device); model.eval()
    model.config.model_update_type = "cut3r_taum_log"

    data_root = "/mnt/sda/szy/research/dataset/tum"
    seqs = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])[:args.max_seqs]

    all_seq_stats = []
    for seq in seqs:
        rgb_txt = os.path.join(data_root, seq, "rgb.txt")
        if os.path.exists(rgb_txt):
            with open(rgb_txt) as f:
                lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
            filelist = [os.path.join(data_root, seq, l.split()[-1]) for l in lines]
        else:
            filelist = sorted([os.path.join(data_root, seq, "rgb", f) for f in os.listdir(os.path.join(data_root, seq, "rgb")) if f.endswith('.png')])
        filelist = filelist[:args.max_frames]
        if len(filelist) < 3: continue

        print(f"\nProcessing {seq} ({len(filelist)} frames)...")
        views = prepare_input(filelist, size=args.size)
        model._taum_log = []
        model._taum_prev_new_state = None
        model._taum_prev_feat = None
        with torch.no_grad():
            inference_recurrent_lighter(views, model, device, verbose=False)

        log = model._taum_log
        if log:
            stats = {
                "seq": seq, "n_frames": len(filelist), "n_gated": len(log),
                "temporal_mean": float(np.mean([g["temporal_mean"] for g in log])),
                "temporal_std_time": float(np.std([g["temporal_mean"] for g in log])),
                "temporal_std_dim": float(np.mean([g["temporal_std"] for g in log])),
                "spatial_mean": float(np.mean([g["spatial_mean"] for g in log])),
                "spatial_std_time": float(np.std([g["spatial_mean"] for g in log])),
                "final_mean": float(np.mean([g["final_mean"] for g in log])),
                "final_std_time": float(np.std([g["final_mean"] for g in log])),
                "sc_cv": float(np.mean([g["sc_cv"] for g in log])),
            }
            all_seq_stats.append(stats)
            print(f"  temporal: μ={stats['temporal_mean']:.4f}, σ_t={stats['temporal_std_time']:.4f}, σ_d={stats['temporal_std_dim']:.4f}")
            print(f"  spatial:  μ={stats['spatial_mean']:.4f}, σ_t={stats['spatial_std_time']:.4f}")
            print(f"  final:    μ={stats['final_mean']:.4f}, σ_t={stats['final_std_time']:.4f}")
            print(f"  CV: {stats['sc_cv']:.4f}")

    # Summary
    print("\n" + "=" * 110)
    print(f"TAUM Gate Statistics — {args.dataset.upper()} ({len(all_seq_stats)} seqs)")
    print("=" * 110)
    hdr = f"{'Seq':<45} {'Temp μ':>8} {'Temp σ_t':>8} {'Temp σ_d':>8} {'Spat μ':>8} {'Final μ':>8} {'Final σ_t':>9} {'CV':>6}"
    print(hdr); print("-" * 110)
    for s in all_seq_stats:
        print(f"{s['seq']:<45} {s['temporal_mean']:>8.4f} {s['temporal_std_time']:>8.4f} {s['temporal_std_dim']:>8.4f} {s['spatial_mean']:>8.4f} {s['final_mean']:>8.4f} {s['final_std_time']:>9.4f} {s['sc_cv']:>6.3f}")
    if all_seq_stats:
        print("-" * 110)
        a = lambda k: np.mean([s[k] for s in all_seq_stats])
        print(f"{'AVERAGE':<45} {a('temporal_mean'):>8.4f} {a('temporal_std_time'):>8.4f} {a('temporal_std_dim'):>8.4f} {a('spatial_mean'):>8.4f} {a('final_mean'):>8.4f} {a('final_std_time'):>9.4f} {a('sc_cv'):>6.3f}")
        exp = torch.sigmoid(torch.tensor(-0.5)).item()
        print(f"\n=== Key Finding ===")
        print(f"TAUM temporal: μ={a('temporal_mean'):.4f}, expected sigmoid(-0.5)={exp:.4f}")
        print(f"  σ_time={a('temporal_std_time'):.4f} {'→ NEAR-CONSTANT (< 0.02)' if a('temporal_std_time') < 0.02 else ''}")
        print(f"  CV={a('sc_cv'):.4f} {'→ low CV: state_change/mean ≈ 1, gate ≈ constant' if a('sc_cv') < 0.5 else ''}")
        print(f"Final (TAUM×SCUM): μ={a('final_mean'):.4f}, σ_time={a('final_std_time'):.4f}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"dataset": args.dataset, "stats": all_seq_stats}, f, indent=2)
    print(f"\nSaved to {args.output}")

if __name__ == "__main__":
    main()
