"""Quick check: is decoder cross-attention sparse or diffuse?"""
import sys, torch, numpy as np
sys.path.insert(0, 'src')
from dust3r.model import ARCroco3DStereo
from dust3r.blocks import CrossAttention
from PIL import Image
import torchvision.transforms as T
import glob

# Monkey-patch CrossAttention to capture attention weights
cross_attn_weights = []
original_forward = CrossAttention.forward

def patched_forward(self, query, key, value, qpos, kpos, return_attn=False):
    result = original_forward(self, query, key, value, qpos, kpos, return_attn=True)
    x, attn_presoftmax = result
    attn_softmax = torch.softmax(attn_presoftmax, dim=-1)
    cross_attn_weights.append(attn_softmax.detach().cpu())
    return x

CrossAttention.forward = patched_forward

# Load model
print("Loading model...")
model = ARCroco3DStereo.from_pretrained('model/cut3r_512_dpt_4_64.pth').cuda().eval()

# Load images
scene_dir = 'data/long_scannet_s3/scene0707_00/color_1000'
imgs = sorted(glob.glob(f'{scene_dir}/frame_*.jpg'))[:3]
transform = T.Compose([
    T.Resize((368, 512)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

device = 'cuda'
print(f"Loading {len(imgs)} images...")

with torch.no_grad():
    # Encode frame 0 → init state
    img0 = transform(Image.open(imgs[0]).convert('RGB')).unsqueeze(0).to(device)
    true_shape = torch.tensor([[368, 512]], device=device)
    feat0_list, pos0, _ = model._encode_image(img0, true_shape)
    feat0 = feat0_list[0]  # [1, n_patches, D]
    state_feat, state_pos = model._init_state(feat0, pos0)
    init_state_feat = state_feat.clone()

    # Encode frame 1
    img1 = transform(Image.open(imgs[1]).convert('RGB')).unsqueeze(0).to(device)
    feat1_list, pos1, _ = model._encode_image(img1, true_shape)
    feat1 = feat1_list[0]

    # Setup pose
    pose_feat = model.pose_token.expand(1, -1, -1)
    pose_pos = -torch.ones(1, 1, 2, device=device, dtype=pos1.dtype)

    # Run rollout (triggers hooks)
    cross_attn_weights.clear()
    new_state_feat, dec, *_ = model._recurrent_rollout(
        state_feat, state_pos, feat1, pos1,
        pose_feat, pose_pos, init_state_feat,
        return_attn=False
    )

    print(f"Captured {len(cross_attn_weights)} CrossAttention calls")
    # Each DecoderBlock has 2 CrossAttention: state→img (even idx), img→state (odd idx)
    # 12 layers × 2 = 24 total

    state_to_img = [cross_attn_weights[i] for i in range(0, len(cross_attn_weights), 2)]
    print(f"State-to-image layers: {len(state_to_img)}, shape: {state_to_img[0].shape}")

    all_attn = torch.stack(state_to_img, dim=0).squeeze(1)  # [L, H, Ns, Ni]
    L, H, Ns, Ni = all_attn.shape
    print(f"Combined: [{L} layers, {H} heads, {Ns} state, {Ni} img]")

    attn_avg = all_attn.mean(dim=(0, 1))  # [Ns, Ni]
    eps = 1e-10
    max_entropy = np.log(Ni)

    # === 1. Entropy ===
    entropy = -(attn_avg * (attn_avg + eps).log()).sum(dim=-1)
    norm_entropy = entropy.numpy() / max_entropy

    print(f"\n{'='*60}")
    print(f"ATTENTION ENTROPY (0=focused, 1=uniform)")
    print(f"{'='*60}")
    print(f"Mean: {norm_entropy.mean():.4f}")
    print(f"Std:  {norm_entropy.std():.4f}")
    print(f"Min:  {norm_entropy.min():.4f}")
    print(f"Max:  {norm_entropy.max():.4f}")
    print(f"% tokens entropy < 0.5: {(norm_entropy < 0.5).mean()*100:.1f}%")
    print(f"% tokens entropy < 0.8: {(norm_entropy < 0.8).mean()*100:.1f}%")
    print(f"% tokens entropy > 0.95: {(norm_entropy > 0.95).mean()*100:.1f}%")

    # === 2. Effective patches ===
    eff_patches = torch.exp(entropy)
    print(f"\n{'='*60}")
    print(f"EFFECTIVE ATTENDED PATCHES (out of {Ni})")
    print(f"{'='*60}")
    print(f"Mean: {eff_patches.mean().item():.1f}")
    print(f"Median: {eff_patches.median().item():.1f}")
    print(f"Min: {eff_patches.min().item():.1f}")
    print(f"Max: {eff_patches.max().item():.1f}")

    # === 3. Diversity ===
    attn_norm = attn_avg / (attn_avg.norm(dim=-1, keepdim=True) + eps)
    torch.manual_seed(42)
    idx = torch.randint(0, Ns, (200,))
    cosine_sim = (attn_norm[idx[:100]] * attn_norm[idx[100:]]).sum(dim=-1)

    print(f"\n{'='*60}")
    print(f"ATTENTION DIVERSITY (cosine sim between random token pairs)")
    print(f"{'='*60}")
    print(f"Mean: {cosine_sim.mean().item():.4f}")
    print(f"Std:  {cosine_sim.std().item():.4f}")
    print(f"(1.0 = identical, 0.0 = different)")

    # === 4. Spatial centers ===
    has_pose = (Ni == 577)
    patch_attn = attn_avg[:, 1:] if has_pose else attn_avg
    n_patches = patch_attn.shape[-1]
    grid = int(np.sqrt(n_patches))
    patch_2d = patch_attn.reshape(Ns, grid, grid)
    yc = torch.arange(grid).float()
    xc = torch.arange(grid).float()
    attn_y = (patch_2d.sum(-1) * yc[None]).sum(-1) / (patch_2d.sum((-1,-2)) + eps)
    attn_x = (patch_2d.sum(-2) * xc[None]).sum(-1) / (patch_2d.sum((-1,-2)) + eps)

    print(f"\n{'='*60}")
    print(f"SPATIAL ATTENTION CENTERS (grid 0-{grid-1})")
    print(f"{'='*60}")
    print(f"Y: mean={attn_y.mean():.2f}, std={attn_y.std():.2f}")
    print(f"X: mean={attn_x.mean():.2f}, std={attn_x.std():.2f}")

    # === 5. Per-layer ===
    print(f"\n{'='*60}")
    print(f"PER-LAYER ENTROPY")
    print(f"{'='*60}")
    for l in range(L):
        la = all_attn[l].mean(0)
        le = -(la * (la + eps).log()).sum(-1)
        print(f"  Layer {l:2d}: {(le / max_entropy).mean().item():.4f}")

    # === VERDICT ===
    me = norm_entropy.mean()
    mc = cosine_sim.mean().item()
    ss = (attn_y.std().item() + attn_x.std().item()) / 2

    print(f"\n{'='*60}")
    print(f"VERDICT")
    print(f"{'='*60}")
    print(f"Entropy={me:.4f}, Cosine={mc:.4f}, SpatialStd={ss:.2f}")
    if me > 0.9 and mc > 0.9:
        print("=> DIFFUSE. Route C will NOT work.")
    elif me > 0.85 or mc > 0.85:
        print("=> MOSTLY DIFFUSE. Route C RISKY.")
    elif ss < 2.0:
        print("=> LOW SPATIAL DIVERSITY. Route C RISKY.")
    else:
        print("=> SPARSE & DIVERSE. Route C has POTENTIAL!")
