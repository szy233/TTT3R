"""Analyze per-frame adaptive signals (drift energy, cos_sim, delta norms, etc.)
across different scenes/datasets to inform adaptive strategy design.

Usage:
    CUDA_VISIBLE_DEVICES=0 python analysis/analyze_adaptive_signals.py
"""
import os
import sys
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.dust3r.model import ARCroco3DStereo
from src.dust3r.inference import inference_recurrent

# Monkey-patch _delta_ortho_update to collect per-frame stats
_original_ortho = ARCroco3DStereo._delta_ortho_update

# Global collector
frame_stats = []

def _patched_ortho_update(state_feat, new_state_feat, ortho_state, config):
    """Wrapper that collects stats before calling original."""
    result = _original_ortho(state_feat, new_state_feat, ortho_state, config)

    step = ortho_state.get('step', 0)
    ema_drift_e = ortho_state.get('ema_drift_energy', None)

    if ema_drift_e is not None:
        de = ema_drift_e.detach()
        delta = new_state_feat - state_feat
        delta_norm = delta.norm(dim=-1)  # [B, T]

        frame_stats.append({
            'step': step,
            'drift_energy_mean': de.mean().item(),
            'drift_energy_std': de.std().item(),
            'drift_energy_min': de.min().item(),
            'drift_energy_max': de.max().item(),
            'drift_energy_q25': de.quantile(0.25).item(),
            'drift_energy_q75': de.quantile(0.75).item(),
            'delta_norm_mean': delta_norm.mean().item(),
            'delta_norm_std': delta_norm.std().item(),
            'frac_de_gt_03': (de > 0.3).float().mean().item(),
            'frac_de_gt_05': (de > 0.5).float().mean().item(),
            'frac_de_gt_07': (de > 0.7).float().mean().item(),
        })

    return result

ARCroco3DStereo._delta_ortho_update = staticmethod(_patched_ortho_update)


def load_model(weights_path, device):
    model = ARCroco3DStereo.from_pretrained(weights_path)
    model.config.model_update_type = 'ddd3r'
    model.config.alpha_perp = 0.5
    model.config.alpha_parallel = 0.05
    model.config.beta_ema = 0.95
    model.config.gamma = 0.0  # pure ortho to see raw signals
    model.config.auto_gamma = ''
    model = model.to(device)
    return model


def get_scene_images(dataset_name, scene_name):
    """Get image file list for a scene."""
    from eval.relpose.metadata import dataset_metadata
    cfg = dataset_metadata[dataset_name]

    img_path = cfg['img_path']
    dir_path = cfg['dir_path_func'](img_path, scene_name)

    if not os.path.exists(dir_path):
        return None

    imgs = sorted(os.listdir(dir_path))
    img_list = [os.path.join(dir_path, f) for f in imgs if f.endswith(('.jpg', '.png'))]
    return img_list


def run_scene(model, dataset_name, scene_name, device):
    """Run inference on one scene and return collected stats."""
    global frame_stats
    frame_stats = []

    img_list = get_scene_images(dataset_name, scene_name)
    if img_list is None or len(img_list) < 3:
        print(f"  Skip {scene_name}: not enough images")
        return None

    from src.dust3r.utils.image import load_images
    from src.dust3r.inference import inference_recurrent_lighter

    images = load_images(img_list, size=512, verbose=False)
    views = []
    for i in range(len(images)):
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

    with torch.no_grad():
        _ = inference_recurrent_lighter(views, model, device, verbose=False)

    return frame_stats.copy()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model('model/cut3r_512_dpt_4_64.pth', device)

    # Representative scenes
    configs = {
        'tum_90f': {
            'dataset': 'tum',
            'scenes': ['rgbd_dataset_freiburg3_sitting_static',
                       'rgbd_dataset_freiburg3_walking_xyz',
                       'rgbd_dataset_freiburg3_long_office_household'],
        },
        'scannet_90f': {
            'dataset': 'scannet_s3_90',
            'scenes': ['scene0707_00', 'scene0710_00', 'scene0750_00',
                       'scene0770_00', 'scene0800_00'],
        },
        'scannet_1000f': {
            'dataset': 'scannet_s3_1000',
            'scenes': ['scene0707_00', 'scene0710_00'],
        },
        'tum_1000f': {
            'dataset': 'tum_s1_1000',
            'scenes': ['rgbd_dataset_freiburg3_sitting_static',
                       'rgbd_dataset_freiburg3_walking_xyz'],
        },
    }

    all_results = {}

    for group_name, cfg in configs.items():
        print(f"\n=== {group_name} ({cfg['dataset']}) ===")
        group_results = {}
        for scene in cfg['scenes']:
            print(f"  Running {scene}...")
            stats = run_scene(model, cfg['dataset'], scene, device)
            if stats:
                group_results[scene] = stats
                n = len(stats)
                de_values = [s['drift_energy_mean'] for s in stats]
                print(f"    {n} frames, drift_e: mean={np.mean(de_values):.3f}, "
                      f"std={np.std(de_values):.3f}, "
                      f"range=[{np.min(de_values):.3f}, {np.max(de_values):.3f}]")

                frac_gt05 = [s['frac_de_gt_05'] for s in stats]
                frac_gt07 = [s['frac_de_gt_07'] for s in stats]
                print(f"    frac(de>0.5): {np.mean(frac_gt05):.2f}, "
                      f"frac(de>0.7): {np.mean(frac_gt07):.2f}")

        all_results[group_name] = group_results

    # Save raw data
    os.makedirs('analysis/output', exist_ok=True)
    with open('analysis/output/adaptive_signals.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print_summary(all_results)

    # Plot
    plot_results(all_results)
    print("\nSaved to analysis/output/adaptive_signals.json and adaptive_signals.png")


def print_summary(all_results):
    """Print comprehensive summary for adaptive strategy design."""
    print("\n" + "="*80)
    print("ADAPTIVE SIGNAL ANALYSIS SUMMARY")
    print("="*80)

    for group, scenes_data in all_results.items():
        print(f"\n--- {group} ---")
        all_de = []
        all_frac05 = []
        all_frac07 = []
        all_de_std = []

        for scene, stats in scenes_data.items():
            de = [s['drift_energy_mean'] for s in stats]
            de_std = [s['drift_energy_std'] for s in stats]
            frac05 = [s['frac_de_gt_05'] for s in stats]
            frac07 = [s['frac_de_gt_07'] for s in stats]

            all_de.extend(de)
            all_frac05.extend(frac05)
            all_frac07.extend(frac07)
            all_de_std.extend(de_std)

            short_name = scene.split('_')[-1] if 'scene' in scene else '_'.join(scene.split('_')[3:5])
            print(f"  {short_name:20s}: de={np.mean(de):.3f}+/-{np.std(de):.3f}  "
                  f"token_std={np.mean(de_std):.3f}  "
                  f"frac>0.5={np.mean(frac05):.2f}  frac>0.7={np.mean(frac07):.2f}")

        if all_de:
            print(f"  {'OVERALL':20s}: de={np.mean(all_de):.3f}+/-{np.std(all_de):.3f}  "
                  f"token_std={np.mean(all_de_std):.3f}  "
                  f"frac>0.5={np.mean(all_frac05):.2f}  frac>0.7={np.mean(all_frac07):.2f}")

    # Key insights
    print("\n" + "="*80)
    print("KEY INSIGHTS FOR ADAPTIVE DESIGN")
    print("="*80)

    tum_de = []
    sn_de = []
    for group, scenes_data in all_results.items():
        for scene, stats in scenes_data.items():
            de = [s['drift_energy_mean'] for s in stats]
            if 'tum' in group:
                tum_de.extend(de)
            elif 'scannet' in group:
                sn_de.extend(de)

    if tum_de and sn_de:
        # Overlap analysis
        tum_arr = np.array(tum_de)
        sn_arr = np.array(sn_de)
        print(f"\nTUM drift_e distribution:    [{np.percentile(tum_arr,5):.3f}, {np.percentile(tum_arr,95):.3f}] (5-95th pct)")
        print(f"ScanNet drift_e distribution: [{np.percentile(sn_arr,5):.3f}, {np.percentile(sn_arr,95):.3f}] (5-95th pct)")
        overlap_min = max(np.percentile(tum_arr, 5), np.percentile(sn_arr, 5))
        overlap_max = min(np.percentile(tum_arr, 95), np.percentile(sn_arr, 95))
        if overlap_max > overlap_min:
            print(f"Overlap zone: [{overlap_min:.3f}, {overlap_max:.3f}]")
            tum_in_overlap = np.mean((tum_arr >= overlap_min) & (tum_arr <= overlap_max))
            sn_in_overlap = np.mean((sn_arr >= overlap_min) & (sn_arr <= overlap_max))
            print(f"TUM frames in overlap: {tum_in_overlap:.1%}, ScanNet frames in overlap: {sn_in_overlap:.1%}")
        else:
            print("NO overlap in 5-95th percentile → clean separation possible!")


def plot_results(all_results):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    colors_tum = plt.cm.Blues(np.linspace(0.4, 0.9, 5))
    colors_sn = plt.cm.Reds(np.linspace(0.4, 0.9, 5))

    # Plot 1: drift_energy_mean over frames (90f)
    ax = axes[0, 0]
    ci = 0
    for group in ['tum_90f', 'scannet_90f']:
        colors = colors_tum if 'tum' in group else colors_sn
        for i, (scene, stats) in enumerate(all_results.get(group, {}).items()):
            de = [s['drift_energy_mean'] for s in stats]
            short = scene.split('_')[-1] if 'scene' in scene else scene.split('_')[3]
            ax.plot(de, alpha=0.7, color=colors[i % len(colors)], label=f"{group[:3]}/{short}")
    ax.set_xlabel('Frame')
    ax.set_ylabel('Drift Energy (mean over tokens)')
    ax.set_title('Per-frame Drift Energy (90f)')
    ax.legend(fontsize=6, ncol=2)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    # Plot 2: histogram of per-frame drift energy
    ax = axes[0, 1]
    for group, color, label in [('tum_90f', 'blue', 'TUM'), ('scannet_90f', 'red', 'ScanNet')]:
        des = []
        for scene, stats in all_results.get(group, {}).items():
            des.extend([s['drift_energy_mean'] for s in stats])
        if des:
            ax.hist(des, bins=30, alpha=0.5, color=color, label=label, density=True)
    ax.set_xlabel('Drift Energy (mean)')
    ax.set_ylabel('Density')
    ax.set_title('Drift Energy Distribution')
    ax.legend()
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)

    # Plot 3: Per-token DE spread (IQR)
    ax = axes[0, 2]
    for group in ['tum_90f', 'scannet_90f']:
        colors = colors_tum if 'tum' in group else colors_sn
        for i, (scene, stats) in enumerate(list(all_results.get(group, {}).items())[:2]):
            q25 = [s['drift_energy_q25'] for s in stats]
            q75 = [s['drift_energy_q75'] for s in stats]
            mean = [s['drift_energy_mean'] for s in stats]
            short = scene.split('_')[-1] if 'scene' in scene else scene.split('_')[3]
            x = range(len(mean))
            ax.fill_between(x, q25, q75, alpha=0.15, color=colors[i])
            ax.plot(mean, color=colors[i], label=f"{group[:3]}/{short}")
    ax.set_xlabel('Frame')
    ax.set_ylabel('Drift Energy')
    ax.set_title('Per-token DE spread (IQR)')
    ax.legend(fontsize=7)

    # Plot 4: frac(de>0.5) per frame
    ax = axes[1, 0]
    for group in ['tum_90f', 'scannet_90f']:
        colors = colors_tum if 'tum' in group else colors_sn
        for i, (scene, stats) in enumerate(all_results.get(group, {}).items()):
            frac = [s['frac_de_gt_05'] for s in stats]
            short = scene.split('_')[-1] if 'scene' in scene else scene.split('_')[3]
            ax.plot(frac, alpha=0.7, color=colors[i % len(colors)], label=f"{group[:3]}/{short}")
    ax.set_xlabel('Frame')
    ax.set_ylabel('Fraction of tokens')
    ax.set_title('Fraction tokens with DE > 0.5')
    ax.legend(fontsize=6, ncol=2)

    # Plot 5: delta norm
    ax = axes[1, 1]
    for group in ['tum_90f', 'scannet_90f']:
        colors = colors_tum if 'tum' in group else colors_sn
        for i, (scene, stats) in enumerate(all_results.get(group, {}).items()):
            dn = [s['delta_norm_mean'] for s in stats]
            short = scene.split('_')[-1] if 'scene' in scene else scene.split('_')[3]
            ax.plot(dn, alpha=0.7, color=colors[i % len(colors)], label=f"{group[:3]}/{short}")
    ax.set_xlabel('Frame')
    ax.set_ylabel('Delta Norm (mean)')
    ax.set_title('Update Magnitude')
    ax.legend(fontsize=6, ncol=2)

    # Plot 6: 1000f drift energy
    ax = axes[1, 2]
    for group in ['tum_1000f', 'scannet_1000f']:
        colors = colors_tum if 'tum' in group else colors_sn
        for i, (scene, stats) in enumerate(all_results.get(group, {}).items()):
            de = [s['drift_energy_mean'] for s in stats]
            short = scene.split('_')[-1] if 'scene' in scene else scene.split('_')[3]
            ax.plot(de, alpha=0.7, color=colors[i % len(colors)], label=f"{group[:5]}/{short}")
    ax.set_xlabel('Frame')
    ax.set_ylabel('Drift Energy (mean)')
    ax.set_title('Drift Energy (1000f)')
    ax.legend(fontsize=7)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('analysis/output/adaptive_signals.png', dpi=150)
    plt.close()


if __name__ == '__main__':
    main()
