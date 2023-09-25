"""
    General useful utility functions for image processing
"""


import gputils.startup_guyga as gputils
from skimage import io as skio
import skvideo.io  
from imutils import build_montages
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from matplotlib import animation
import torchvision.io
from scipy import signal as sig

    
def make_montage(imgs_or_image_paths, n_col=None, return_pil=False, **get_images_kws):
    if isinstance(imgs_or_image_paths[0], np.ndarray):
        imgs = imgs_or_image_paths
    else:
        imgs = get_images(imgs_or_image_paths, use_pil=True, force_rgb=True, **get_images_kws)    
    N = len(imgs)
    im_res = imgs[0].shape[0]
    if n_col is None:
        n_col = int(np.sqrt(N) * 4 // 3)
    n_col = min(n_col, N)
    image_montage = build_montages(imgs, (im_res, im_res), (n_col, N // n_col))[0]
    if return_pil:
        return Image.fromarray(image_montage)
    return image_montage


def get_images(images_paths, resize_dim=None, n_threads=20, use_pil=False, force_rgb=False, reported=False, 
               return_list=False):
    """Return the list of frames given by list of absolute paths.
    """
    if resize_dim is None:
        resize_fn = lambda x: x
    else:
        if use_pil:
            resize_fn = lambda x: x.resize(resize_dim, Image.ANTIALIAS)
        else:
            resize_fn = lambda x: skio.transform.resize(x, resize_dim, anti_aliasing=True)
    if use_pil:
        if force_rgb:
            reader_fn = lambda image_path: np.array(resize_fn(Image.open(image_path).convert('RGB')))
        else:
            reader_fn = lambda image_path: np.array(resize_fn(Image.open(image_path)))
    else:
        if force_rgb:
            raise NotImplementedError
        reader_fn = lambda image_path: resize_fn(skio.imread(image_path))
    if reported:
        MyPool = gputils.PoolReported
    else:
        MyPool = gputils.Pool
    with MyPool(n_threads) as pool:
        res_list = pool.map(reader_fn, images_paths)
    if return_list:
        return res_list
    else:
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

    
def add_header(imgs, labels, margin=40, side='top', fontsize=14):
    if not isinstance(labels, list):
        labels = [labels] * len(imgs)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", fontsize)
    img_list = []
    for im, lbl in zip(imgs, labels):
        im_margin_pil = pilify(np.full_like(im, 255, dtype='uint8')[:margin])
        draw = ImageDraw.Draw(im_margin_pil)
        _, _, w, h = draw.textbbox((0, 0), lbl, font=font)
        draw.text(((im_margin_pil.size[0]-w)/2, (margin-h)/2), lbl , (0, 0 ,0), font=font)
        im_margin = np.array(im_margin_pil)
        if side == 'top':
            im = np.vstack([im_margin, im])
        elif side == 'bottom':
            im = np.vstack([im, im_margin])
        img_list.append(np.array(im))
    return img_list


def interp_images(im1, im2, alpha=None):
    if alpha is None:
        alpha = np.random.rand()
    interpolated = (alpha * im1 + (1-alpha) * im2)
    if isinstance(interpolated, np.ndarray):
        return interpolated.astype(im1.dtype)
    else:
        return interpolated.type(im1.dtype)


def get_video(video_path, resize_dim=None, reader_fn=skvideo.io.vread):
    """Return array of NxHxWx3 of frames given by the video path.
    """
    vid = reader_fn(video_path)
    if resize_dim is None:
        return vid
    else:
        return resize_vid(vid, resize_dim)
    

def resize_vid(vid, resize_dim):
    """Assume vid is NxHxWx3
    """
    return np.stack([np.array(Image.fromarray(vid[i, :, :, :])
                              .resize(resize_dim, Image.ANTIALIAS)) 
                     for i in range(vid.shape[0])])


def html_vid(vid, interval=100):
    """
        Use in jupyter:
        anim = html_vid(q_vid)
        HTML(anim.to_html5_video())
    """
    fig = gputils.plt.figure()
    fig.tight_layout()
    im = gputils.plt.imshow(vid[0, :, :, :])
    gputils.plt.axis('off')
    gputils.plt.close()  # this is required to not display the generated image

    def init():
        im.set_data(vid[0, :, :, :])

    def animate(i):
        im.set_data(vid[i, :, :, :])
        return im

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=vid.shape[0], interval=interval)
    return anim


def write_video(vid, video_path, fps: int=15, **kwargs):
    torchvision.io.write_video(video_path, vid, fps, **kwargs)

# def save_gif(frames_pil, filepath, duration=100, loop=1):
#     """Images don't look good"""
#     frames = [gputils.Image.fromarray(arr) for arr in frames_pil]
#     frames[0].save(filepath, format="GIF", append_images=frames, 
#                    save_all=True, duration=duration, loop=loop)


def butter_lowpass(cutoff, fs, order=5):
    return sig.butter(order, cutoff, fs=fs, btype='low', analog=False)


def butter_lowpass_filter(data, cutoff, fs, order=6):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = sig.lfilter(b, a, data)
    return y
