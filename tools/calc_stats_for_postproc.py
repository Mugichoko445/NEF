import argparse
from pathlib import Path
import numpy as np
import cv2


def calc_stats(img, lab):
    # mask out complete gray pixels
    gray_mask = np.logical_and(img[:,:,0] == img[:,:,1], img[:,:,0] == img[:,:,2])
    px = lab[np.logical_not(gray_mask)]
    return np.mean(px[..., 1]), np.mean(px[..., 2]), np.std(px[..., 1]), np.std(px[..., 2])
    

def calc_stats_for_postproc(images_dir:Path, image_format:str="png"):
    images = sorted(images_dir.glob("*." + image_format))

    image_stats = []
    for image in images:
        image = cv2.imread(str(image), cv2.IMREAD_COLOR)
        image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # mean of a, mean of b, std of a, and std of b
        image_stats.append(calc_stats(image, image_lab))

    means_med_a = np.median(np.array([v for (v, _, _, _) in image_stats]))
    means_med_b = np.median(np.array([v for (_, v, _, _) in image_stats]))
    stds_med_a = np.median(np.array([v for (_, _, v, _) in image_stats]))
    stds_med_b = np.median(np.array([v for (_, _, _, v) in image_stats]))

    return np.array([means_med_a, means_med_b, stds_med_a, stds_med_b])


def parse_args():
    parser = argparse.ArgumentParser(description="calc stats for post-proc")
    parser.add_argument("renderings_dir", type=Path, help="path to a renderings directory")
    parser.add_argument("--image-format", type=str, default="png")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    stats = calc_stats_for_postproc(args.renderings_dir, args.image_format)

    np.save(args.renderings_dir.parent / Path(args.renderings_dir.name + "_stats.npy"), stats)
