import argparse
from pathlib import Path
import shutil
import numpy as np
import cv2


def create_diff_images(dataset_dir: Path, renderings_dir: Path, downscale_factor:int = 1, thresh_ratio: float = 0.1, colormap: int = cv2.COLORMAP_VIRIDIS, image_format: str = "png"):
    images_dir = Path("images" if downscale_factor == 1 else "images_" + str(downscale_factor))
    dst_dir = renderings_dir.parent / images_dir
    dst_dir.mkdir(exist_ok=True)
    dst_sup_dir = renderings_dir.parent / Path("absdiff_images")
    dst_sup_dir.mkdir(exist_ok=True)

    images = sorted((dataset_dir / images_dir).glob("*." + image_format))
    renderings = sorted(renderings_dir.glob("*." + image_format))
    assert len(images) == len(renderings), f"inconsistent input files ({len(images)} != {len(renderings)})"

    # find the maximum sum of absolute diff (sad) values
    max_sads = []
    for image, rendering in zip(images, renderings):
        img_gt = cv2.imread(str(image), cv2.IMREAD_COLOR)
        img_rendering = cv2.imread(str(rendering), cv2.IMREAD_COLOR)

        sad = np.sum(np.abs(img_gt.astype(np.int32) - img_rendering.astype(np.int32)), axis=2)
        max_sads.append(np.max(sad))
    max_sad = np.max(max_sads)

    # create the final diff image and save it
    for image, rendering in zip(images, renderings):
        img_gt = cv2.imread(str(image), cv2.IMREAD_COLOR)
        img_gt_gray = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
        img_rendering = cv2.imread(str(rendering), cv2.IMREAD_COLOR)

        img_absdiff = np.abs(img_gt.astype(np.int32) - img_rendering.astype(np.int32))
        sad = np.sum(img_absdiff, axis=2)
        norm_sad = sad / max_sad
        mask = norm_sad > (thresh_ratio * np.max(norm_sad))

        img_absdiff = img_absdiff.astype(np.uint8)
        img_absdiff_gray = cv2.cvtColor(img_absdiff, cv2.COLOR_BGR2GRAY)
        img_colored = cv2.applyColorMap(img_absdiff_gray, colormap)

        img_diff = np.where(np.dstack([mask] * 3), img_colored, np.dstack([img_gt_gray] * 3))

        cv2.imwrite(str(dst_dir / image.name), img_diff)
        cv2.imwrite(str(dst_sup_dir / image.name), img_absdiff)


def copy_transforms_json(dataset_dir: Path, diff_dataset_dir: Path, transforms_json: Path = "transforms.json"):
    dst_dir = diff_dataset_dir.parent
    dst_dir.mkdir(exist_ok=True)

    src_filename = dataset_dir / transforms_json
    if src_filename.exists:
        dst_filename = dst_dir / src_filename.name
        shutil.copy(src_filename, dst_filename)


def parse_args():
    parser = argparse.ArgumentParser(description="create camera_path.json")
    parser.add_argument("dataset_dir", type=Path, help="path to a director that contains 'images'")
    parser.add_argument("renderings_dir", type=Path, help="path to a director of 'renderings'")
    parser.add_argument("--downscale-factor", type=int, default=1, help="downscale factor used for ns-train and ns-render")
    parser.add_argument("--thresh_ratio", type=float, default=0.1, help="threshold value for max absolute diff")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_diff_images(args.dataset_dir, args.renderings_dir, args.downscale_factor, thresh_ratio=args.thresh_ratio, colormap=cv2.COLORMAP_VIRIDIS)
    copy_transforms_json(args.dataset_dir, args.renderings_dir)
