import argparse
import glob
import os
import shutil
from pathlib import Path


DEFAULT_TARGET_FRAMES = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare long KITTI video-depth subsets for TTT3R evaluation."
    )
    parser.add_argument(
        "--source_root",
        type=str,
        required=True,
        help="Root containing KITTI val sequences with proj_depth/groundtruth/image_02.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="./data/long_kitti_s1",
        help="Output root for gathered long KITTI subsets.",
    )
    parser.add_argument(
        "--target_frames",
        type=int,
        nargs="+",
        default=DEFAULT_TARGET_FRAMES,
        help="Target sequence lengths to export.",
    )
    return parser.parse_args()


def find_depth_dirs(source_root):
    pattern = os.path.join(
        source_root, "*", "proj_depth", "groundtruth", "image_02"
    )
    return sorted(glob.glob(pattern))


def infer_image_dir(depth_dir):
    depth_path = Path(depth_dir)
    seq_root = depth_path.parents[3]
    stem_parts = seq_root.name.split("_")
    raw_seq_name = "_".join(stem_parts[:3]) if len(stem_parts) >= 3 else seq_root.name
    return seq_root.parents[0] / raw_seq_name / "image_02" / "data"


def copy_subset(depth_dir, output_root, target_frames):
    depth_path = Path(depth_dir)
    seq_name = depth_path.parents[3].name
    output_root = Path(output_root)
    depth_out = (
        output_root
        / "depth_selection"
        / "val_selection_cropped"
        / f"groundtruth_depth_gathered_{target_frames}"
        / f"{seq_name}_02"
    )
    image_out = (
        output_root
        / "depth_selection"
        / "val_selection_cropped"
        / f"image_gathered_{target_frames}"
        / f"{seq_name}_02"
    )
    depth_out.mkdir(parents=True, exist_ok=True)
    image_out.mkdir(parents=True, exist_ok=True)

    all_depth_files = sorted(depth_path.glob("*.png"))
    actual_frames = min(len(all_depth_files), target_frames)
    print(
        f"sequence {seq_name}: target={target_frames}, "
        f"available={len(all_depth_files)}, processed={actual_frames}"
    )

    image_dir = infer_image_dir(depth_dir)
    for depth_file in all_depth_files[:target_frames]:
        shutil.copy(depth_file, depth_out / depth_file.name)
        image_file = image_dir / depth_file.name
        if image_file.exists():
            shutil.copy(image_file, image_out / image_file.name)
        else:
            print(f"missing image: {image_file}")


def main():
    args = parse_args()
    depth_dirs = find_depth_dirs(args.source_root)
    if not depth_dirs:
        raise FileNotFoundError(
            "No KITTI depth directories found under "
            f"{args.source_root}. Expected */proj_depth/groundtruth/image_02."
        )

    for target_frames in args.target_frames:
        for depth_dir in depth_dirs:
            copy_subset(depth_dir, args.output_root, target_frames)


if __name__ == "__main__":
    main()
