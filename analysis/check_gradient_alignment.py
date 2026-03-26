"""Quick check: does cosine similarity between consecutive state deltas vary meaningfully?"""
import sys, torch, numpy as np
sys.path.insert(0, 'src')
from dust3r.model import ARCroco3DStereo
from PIL import Image
import torchvision.transforms as T
import glob

print("Loading model...")
model = ARCroco3DStereo.from_pretrained('model/cut3r_512_dpt_4_64.pth').cuda().eval()

scene_dir = 'data/long_scannet_s3/scene0707_00/color_1000'
imgs = sorted(glob.glob(f'{scene_dir}/frame_*.jpg'))[:50]
transform = T.Compose([
    T.Resize((368, 512)), T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

device = 'cuda'
true_shape = torch.tensor([[368, 512]], device=device)

frame_cosines = []
frame_cosine_stds = []
frame_gate_means = []

with torch.no_grad():
    # Init
    img0 = transform(Image.open(imgs[0]).convert('RGB')).unsqueeze(0).to(device)
    feat0, pos0, _ = model._encode_image(img0, true_shape)
    feat0 = feat0[0]
    state_feat, state_pos = model._init_state(feat0, pos0)
    init_state_feat = state_feat.clone()
    pose_feat = model.pose_token.expand(1, -1, -1)
    pose_pos = -torch.ones(1, 1, 2, device=device, dtype=state_pos.dtype)

    prev_delta = None

    for i in range(1, len(imgs)):
        img = transform(Image.open(imgs[i]).convert('RGB')).unsqueeze(0).to(device)
        feat, pos, _ = model._encode_image(img, true_shape)
        feat = feat[0]

        # Rollout (just use pose_token every time — we only care about state delta)
        result = model._recurrent_rollout(
            state_feat, state_pos, feat, pos,
            pose_feat, pose_pos, init_state_feat,
            return_attn=False
        )
        new_state_feat = result[0]
        delta = new_state_feat - state_feat  # [1, 768, 768]

        if prev_delta is not None:
            cos = torch.nn.functional.cosine_similarity(
                delta.squeeze(0), prev_delta.squeeze(0), dim=-1
            )  # [768]
            frame_cosines.append(cos.mean().item())
            frame_cosine_stds.append(cos.std().item())
            gate = torch.sigmoid(2.0 * cos)
            frame_gate_means.append(gate.mean().item())

        prev_delta = delta.clone()
        state_feat = new_state_feat  # cut3r mode

        if i % 10 == 0:
            c = frame_cosines[-1] if frame_cosines else 0
            print(f"  Frame {i}: cosine={c:.4f}")

cosines = np.array(frame_cosines)
gates = np.array(frame_gate_means)
cos_stds = np.array(frame_cosine_stds)

print(f"\n{'='*60}")
print(f"GRADIENT ALIGNMENT (cosine between consecutive deltas)")
print(f"{'='*60}")
print(f"Frames: {len(cosines)}")
print(f"Mean cosine:  {cosines.mean():.4f}")
print(f"Std cosine:   {cosines.std():.4f}")
print(f"Min:          {cosines.min():.4f}")
print(f"Max:          {cosines.max():.4f}")
print(f"Range:        {cosines.max() - cosines.min():.4f}")

print(f"\nToken-level std (within frame): {cos_stds.mean():.4f}")

print(f"\n{'='*60}")
print(f"GATE VALUES (sigmoid(2 * cosine))")
print(f"{'='*60}")
print(f"Mean: {gates.mean():.4f}, Std: {gates.std():.4f}")
print(f"Min:  {gates.min():.4f}, Max: {gates.max():.4f}")

print(f"\n{'='*60}")
print(f"TEMPORAL PATTERN")
print(f"{'='*60}")
for i in range(min(15, len(cosines))):
    bar = '#' * int(gates[i] * 40)
    print(f"  Frame {i+2:3d}: cos={cosines[i]:+.4f}  gate={gates[i]:.4f}  |{bar}")
if len(cosines) > 20:
    print(f"  ...")
    for i in range(max(15, len(cosines)-5), len(cosines)):
        bar = '#' * int(gates[i] * 40)
        print(f"  Frame {i+2:3d}: cos={cosines[i]:+.4f}  gate={gates[i]:.4f}  |{bar}")

print(f"\n{'='*60}")
print(f"VERDICT")
print(f"{'='*60}")
if cosines.std() < 0.02:
    print(f"=> CONSTANT. Momentum gate will degenerate.")
elif cosines.std() < 0.05:
    print(f"=> LOW VARIANCE. Limited benefit over constant dampening.")
else:
    print(f"=> MEANINGFUL VARIANCE (std={cosines.std():.4f})!")
    if cos_stds.mean() > 0.05:
        print(f"   Token-level variance also present (std={cos_stds.mean():.4f}) → token-level gating viable!")
    else:
        print(f"   But low token-level variance → frame-level scalar gate only.")
