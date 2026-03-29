"""Preprocess TUM short sequences (90 frames) matching CUT3R/TTSA3R eval protocol.

Source: /mnt/sda/szy/research/dataset/tum/{seq}/rgb/ + groundtruth.txt
Target: data/tum/{seq}/rgb_90/ + groundtruth_90.txt

Usage:
    python datasets_preprocess/prepare_tum_short.py
"""
import glob
import os
import shutil


def read_file_list(filename):
    with open(filename) as f:
        data = f.read()
    lines = data.replace(",", " ").replace("\t", " ").split("\n")
    lst = [
        [v.strip() for v in line.split(" ") if v.strip() != ""]
        for line in lines
        if len(line) > 0 and line[0] != "#"
    ]
    lst = [(float(l[0]), l[1:]) for l in lst if len(l) > 1]
    return dict(lst)


def associate(first_list, second_list, offset=0.0, max_difference=0.02):
    first_keys = set(first_list.keys())
    second_keys = set(second_list.keys())
    potential_matches = [
        (abs(a - (b + offset)), a, b)
        for a in first_keys
        for b in second_keys
        if abs(a - (b + offset)) < max_difference
    ]
    potential_matches.sort()
    matches = []
    for diff, a, b in potential_matches:
        if a in first_keys and b in second_keys:
            first_keys.remove(a)
            second_keys.remove(b)
            matches.append((a, b))
    matches.sort()
    return matches


TUM_ROOT = "/mnt/sda/szy/research/dataset/tum"
OUTPUT_BASE = "./data/tum"
TARGET_FRAMES = 90

os.makedirs(OUTPUT_BASE, exist_ok=True)

dirs = sorted(glob.glob(f"{TUM_ROOT}/*/"))
print(f"Found {len(dirs)} TUM sequences")

for d in dirs:
    first_file = os.path.join(d, "rgb.txt")
    second_file = os.path.join(d, "groundtruth.txt")

    if not os.path.exists(first_file) or not os.path.exists(second_file):
        print(f"  Skipping {d}: missing rgb.txt or groundtruth.txt")
        continue

    first_list = read_file_list(first_file)
    second_list = read_file_list(second_file)
    matches = associate(first_list, second_list)

    frames = []
    gt = []
    for a, b in matches:
        frames.append(os.path.join(d, first_list[a][0]))
        gt.append([b] + second_list[b])

    frames = frames[:TARGET_FRAMES]
    gt_sampled = gt[:TARGET_FRAMES]

    dir_name = os.path.basename(os.path.dirname(d))
    new_dir = os.path.join(OUTPUT_BASE, dir_name, f"rgb_{TARGET_FRAMES}")

    os.makedirs(new_dir, exist_ok=True)
    for frame in frames:
        dst = os.path.join(new_dir, os.path.basename(frame))
        if not os.path.exists(dst):
            shutil.copy(frame, new_dir)

    gt_file = os.path.join(OUTPUT_BASE, dir_name, f"groundtruth_{TARGET_FRAMES}.txt")
    with open(gt_file, "w") as f:
        for pose in gt_sampled:
            f.write(f"{' '.join(map(str, pose))}\n")

    print(f"  {dir_name}: {len(frames)} frames -> {new_dir}")

print("Done.")
