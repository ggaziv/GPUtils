"""
    General useful utility functions for image processing
"""


import gputils.startup_guyga as gputils
from skimage import io as skio
from imutils import build_montages
import numpy as np


def make_montage(imgs, n_col=None):
    N = len(imgs)
    im_res = imgs[0].shape[0]
    if n_col is None:
        n_col = int(np.sqrt(N) * 4 // 3)
    n_col = min(n_col, N)
    return build_montages(imgs, (im_res, im_res), (n_col, N // n_col))[0]


def get_images(images_paths, n_threads=20):
    """Return the list of frames given by list of absolute paths.
    """
    with gputils.PoolReported(n_threads) as pool:
        res_list = pool.map(lambda image_path: skio.imread(image_path), images_paths)
    return np.array(res_list)


def color_framed(img_rgb, pad_size, rgb):
    """Pad image with colored frame
    """
    if pad_size == 0:
        return img_rgb
    dtype = img_rgb.dtype
    img = np.pad(img_rgb, (*[[pad_size] * 2]*2, (0,0)))
    img[:pad_size] = img[-pad_size:] = img[:, -pad_size:] = img[:, :pad_size] = rgb
    return img.astype(dtype)