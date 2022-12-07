"""
    General useful utility functions for image processing
"""


import gputils.startup_guyga as gputils
from skimage import io as skio
from imutils import build_montages
import numpy as np
from PIL import Image, ImageFont, ImageDraw


def make_montage(imgs, n_col=None):
    N = len(imgs)
    im_res = imgs[0].shape[0]
    if n_col is None:
        n_col = int(np.sqrt(N) * 4 // 3)
    n_col = min(n_col, N)
    return build_montages(imgs, (im_res, im_res), (n_col, N // n_col))[0]


def get_images(images_paths, n_threads=20, use_pil=False, force_rgb=False, reported=False):
    """Return the list of frames given by list of absolute paths.
    """
    if use_pil:
        if force_rgb:
            reader_fn = lambda image_path: np.array(Image.open(image_path).convert('RGB'))
        else:
            reader_fn = lambda image_path: np.array(Image.open(image_path))
    else:
        if force_rgb:
            raise NotImplementedError
        reader_fn = lambda image_path: skio.imread(image_path)
    if reported:
        MyPool = gputils.PoolReported
    else:
        MyPool = gputils.Pool
    with MyPool(n_threads) as pool:
        res_list = pool.map(reader_fn, images_paths)
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


def pilify(a):
    """Get/convert to a PIL image"""
    if isinstance(a, np.ndarray):
        if a.dtype != 'uint8':
            a = (a * 255).astype('uint8')
    else:
        raise NotImplementedError
    return gputils.Image.fromarray(a)


class FrameLabeler():
    def __init__(self, color=(255,255,255), xy=(5, 5),
                 font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 16)):
        self.font = font
        self.color = color
        self.xy = xy
            
    def __call__(self, img_pil, label):
        img_pil = pilify(img_pil)
        draw = ImageDraw.Draw(img_pil)
        draw.text(self.xy, label , self.color, font=self.font)
        return img_pil
    

def interp_images(im1, im2, alpha=None):
    if alpha is None:
        alpha = np.random.rand()
    interpolated = (alpha * im1 + (1-alpha) * im2)
    if isinstance(interpolated, np.ndarray):
        return interpolated.astype(im1.dtype)
    else:
        return interpolated.type(im1.dtype)


# def save_gif(frames_pil, filepath, duration=100, loop=1):
#     """Images don't look good"""
#     frames = [gputils.Image.fromarray(arr) for arr in frames_pil]
#     frames[0].save(filepath, format="GIF", append_images=frames, 
#                    save_all=True, duration=duration, loop=loop)