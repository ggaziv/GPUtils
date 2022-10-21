"""
    General useful utility functions for image processing
"""


import gputils.startup_guyga as gputils
from skimage import io as skio
from imutils import build_montages


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