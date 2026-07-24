#!/usr/bin/env python3
"""
E4 — Convert ARKitScenes raw scans into the TUM-style layout our relpose eval expects.

Output mirrors data/long_tum_s1/:
    <out>/<video_id>/rgb_<N>/*.png          (symlinks, no data duplication)
    <out>/<video_id>/groundtruth_<N>.txt    (TUM format: ts tx ty tz qx qy qz qw)

Two details that silently corrupt ATE if got wrong, both handled here:

1. POSE DIRECTION. ARKitScenes `.traj` stores *world-to-camera* (the official loader
   names them r_w_to_p / t_w_to_p and returns np.linalg.inv(extrinsics)).
   TUM groundtruth.txt expects *camera-to-world*. We invert. Verified against
   ARKitScenes/threedod/benchmark_scripts/utils/tenFpsDataLoader.py:TrajStringToMatrix.

2. RATE MISMATCH. Images are recorded at ~60 Hz but poses only at ~10 Hz. We drive
   from the pose stream and attach the nearest image within --tol seconds, so every
   evaluated frame has a genuine measured pose (no interpolation).

Usage:
  python rebuttal/scripts/prepare_arkitscenes.py \
      --src rebuttal/external/ARKitScenes_data/raw/Validation \
      --out data/long_arkit_s1 --lengths 1000 --img-asset lowres_wide
"""
import argparse
import glob
import os
import re

import numpy as np


def angle_axis_to_R(aa):
    """Rodrigues. Matches cv2.Rodrigues / the official convert_angle_axis_to_matrix3."""
    theta = np.linalg.norm(aa)
    if theta < 1e-12:
        return np.eye(3)
    k = aa / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def R_to_quat(R):
    """Rotation matrix -> (qx, qy, qz, qw), TUM ordering."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qx, qy, qz, qw])
    return q / np.linalg.norm(q)


def load_traj(path):
    """-> list of (ts, cam_to_world 4x4). Inverts the stored world-to-camera pose."""
    out = []
    with open(path) as f:
        for line in f:
            tok = line.split()
            if len(tok) != 7:
                continue
            ts = float(tok[0])
            aa = np.array([float(tok[1]), float(tok[2]), float(tok[3])])
            tr = np.array([float(tok[4]), float(tok[5]), float(tok[6])])
            w2c = np.eye(4)
            w2c[:3, :3] = angle_axis_to_R(aa)
            w2c[:3, 3] = tr
            out.append((ts, np.linalg.inv(w2c)))   # -> camera-to-world
    return out


def img_timestamp(path):
    """'41069021_304.644.png' -> 304.644"""
    m = re.search(r"_(\d+\.\d+)\.png$", os.path.basename(path))
    return float(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="ARKitScenes raw/<split> dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lengths", type=int, nargs="+", default=[1000])
    ap.add_argument("--img-asset", default="lowres_wide")
    ap.add_argument("--tol", type=float, default=0.05, help="max |ts| gap, seconds")
    ap.add_argument("--stride", type=int, default=1, help="stride over pose stream")
    args = ap.parse_args()

    scenes = sorted(d for d in os.listdir(args.src)
                    if os.path.isdir(os.path.join(args.src, d)))
    print(f"found {len(scenes)} scenes under {args.src}")
    summary = []

    for vid in scenes:
        sdir = os.path.join(args.src, vid)
        traj_p = os.path.join(sdir, "lowres_wide.traj")
        img_d = os.path.join(sdir, args.img_asset)
        if not (os.path.exists(traj_p) and os.path.isdir(img_d)):
            print(f"  {vid}: SKIP (missing traj or {args.img_asset})")
            continue

        poses = load_traj(traj_p)
        imgs = sorted(glob.glob(os.path.join(img_d, "*.png")))
        its = np.array([img_timestamp(p) for p in imgs], dtype=float)
        order = np.argsort(its)
        its, imgs = its[order], [imgs[i] for i in order]

        pairs = []
        for ts, c2w in poses[::args.stride]:
            j = int(np.searchsorted(its, ts))
            cand = [k for k in (j - 1, j) if 0 <= k < len(its)]
            if not cand:
                continue
            k = min(cand, key=lambda k: abs(its[k] - ts))
            if abs(its[k] - ts) <= args.tol:
                pairs.append((ts, imgs[k], c2w))

        n_avail = len(pairs)
        print(f"  {vid}: {len(poses)} poses, {len(imgs)} imgs -> {n_avail} associated")
        summary.append((vid, len(poses), len(imgs), n_avail))

        for N in args.lengths:
            if n_avail < N:
                print(f"      len {N}: only {n_avail} available, skipping")
                continue
            sel = pairs[:N]
            rgb_out = os.path.join(args.out, vid, f"rgb_{N}")
            os.makedirs(rgb_out, exist_ok=True)
            lines = []
            for ts, ip, c2w in sel:
                dst = os.path.join(rgb_out, f"{ts:.6f}.png")
                if not os.path.lexists(dst):
                    os.symlink(os.path.abspath(ip), dst)
                q = R_to_quat(c2w[:3, :3])
                t = c2w[:3, 3]
                lines.append(f"{ts:.6f} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f} "
                             f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}")
            with open(os.path.join(args.out, vid, f"groundtruth_{N}.txt"), "w") as f:
                f.write("\n".join(lines) + "\n")
            print(f"      len {N}: wrote {len(sel)} frames")

    print("\n=== summary ===")
    for vid, npo, nim, na in summary:
        print(f"{vid}  poses={npo:>6}  imgs={nim:>7}  associated={na:>6}")


if __name__ == "__main__":
    main()
