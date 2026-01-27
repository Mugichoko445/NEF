import argparse
from pathlib import Path
import numpy as np
import cv2

from tools.calc_stats_for_postproc import calc_stats


def match_stats(img:np.array, lab:np.array, ref_stats:np.array):
    """
    all inputs are in np.float32
    """
    matched_lab = lab.copy()
    a_mean, b_mean, a_std, b_std = calc_stats(img, lab)
    matched_lab[...,1] = (lab[...,1] - a_mean) * (ref_stats[2] / a_std) + ref_stats[0]
    matched_lab[...,2] = (lab[...,2] - b_mean) * (ref_stats[3] / b_std) + ref_stats[1]
    return matched_lab


def create_colormap_table(colormap:int):
    cm_tab = np.array([idx for idx in range(256)]).reshape(1, 256)
    cm_tab = cv2.applyColorMap(cm_tab.astype(np.uint8), colormap)
    cm_tab_lab = cv2.cvtColor(cm_tab, cv2.COLOR_BGR2LAB)
    cm_tab_lab = cm_tab_lab.reshape(1, 256, 3)
    cm_tab_lab = cm_tab_lab.astype(np.float32)
    return cm_tab_lab


def replace_luminance_with_enhancement(lab:np.array, cm_tab_lab:np.array, thresh_sigma:float=0.9545):
    """
    luminance of lab is replaced with one from the given table,
    but the luminance is weighted to match the appearance.
    
    all inputs are in np.float32
    """
    res_lab = lab.copy()

    # todo??
    # if matched_lab has [0,0], the luminance is 0
    # else

    # calculate distances in the "ab" space
    distances = np.sum(np.abs(cm_tab_lab[...,-2:] - res_lab[...,np.newaxis,-2:]), axis=-1)
    mindist_indices = np.argmin(distances, axis=-1)

    ab = res_lab[...,-2:]
    selected_cm_tab_ab = cm_tab_lab[0, mindist_indices, -2:]
    ab_norm = np.linalg.norm(ab, axis=-1)
    selected_cm_tab_ab_norm = np.linalg.norm(selected_cm_tab_ab, axis=-1)

    # take dot() / l2-norm, i.e., angle of [0, 1]
    weights = np.sum(ab * selected_cm_tab_ab, axis=-1) / (ab_norm * selected_cm_tab_ab_norm)
    weights[weights <= thresh_sigma] = 0
    res_lab[..., 0] = weights * cm_tab_lab[0, mindist_indices, 0]

    return res_lab


def create_grayish_pixel_mask(img:np.array, thresh:int=15):
    mask_bg = np.abs(img[..., 0] - img[..., 1]) < thresh
    mask_gr = np.abs(img[..., 1] - img[..., 2]) < thresh
    mask_rb = np.abs(img[..., 2] - img[..., 0]) < thresh
    return np.logical_and(np.logical_and(mask_bg, mask_gr), mask_rb)


def postproc(images_dir:Path, ref_stats:np.array, output_suffix:str="neff", image_format:str="png", masking_thresh:int=15, colormap: int = cv2.COLORMAP_VIRIDIS):
    images = sorted(images_dir.glob("*." + image_format))
    dst_dir = images_dir.parent / Path(images_dir.name + output_suffix)
    dst_dir.mkdir(exist_ok=True)

    # colormap table; cm_tab
    cm_tab_lab = create_colormap_table(colormap)

    for image in images:
        img = cv2.imread(str(image), cv2.IMREAD_COLOR)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        img = img.astype(np.float32)
        lab = lab.astype(np.float32)

        matched_lab = match_stats(img, lab, ref_stats)
        #matched_lab[...,0] = 0 # remove luminance (but why??)

        ench_lab = replace_luminance_with_enhancement(matched_lab, cm_tab_lab)
        ench_img = cv2.cvtColor(ench_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        ench_img = ench_img.astype(np.float32)

        # mask out grayish pixels
        mask = create_grayish_pixel_mask(ench_img, masking_thresh)
        ench_img[mask] = 0

        cv2.imwrite(str(dst_dir / image.name), ench_img.astype(np.uint8))


def parse_args():
    parser = argparse.ArgumentParser(description="post-processing")
    parser.add_argument("stats_npy", type=Path, help="stats.npy")
    parser.add_argument("images_dir", type=Path, help="directory containing images to be processed")
    parser.add_argument("--masking-thresh", type=int, default=15)
    parser.add_argument("--output-suffix", type=str, default="_neff")
    parser.add_argument("--image-format", type=str, default="png")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    stats = np.load(args.stats_npy)
    postproc(args.images_dir, stats, args.output_suffix, args.image_format, args.masking_thresh)
