#!/usr/bin/env python
"""
    General useful utility functions
    Date created: 17/4/18
    Python Version: 3.5
"""

import os, shutil, pickle, sys, random, platform
from tkinter import Tk, Label, Entry
from time import localtime, strftime, time
from typing import Iterable
from termcolor import cprint
import logging
import glob
import numpy as np
from matplotlib import pyplot as plt
from warnings import warn
# from tensorflow.python.client import device_lib
import itertools
import contextlib
identity = lambda x: x

__author__ = "Guy Gaziv"

def report(func, msg=''):
    '''
    Decorator function to print a message before execution of a function and report its execution time
    Usage:
        <func_output> = reported(<func>, <msg>)(<func_args>)
    '''
    def reported(*args, **kwargs):
        cprint(msg, 'magenta')
        t1 = time()
        res = func(*args, **kwargs)
        cprint('Completed in %s.' % get_sensible_duration(time()-t1), 'magenta')
        return res
    return reported

class ProgressMonitor:
    '''
    Monitor of progress for iterative procedures including percanrage and remaining time estimates.
    Usage:
        prog_mon = ProgressMonitor(<n_iter>)
        for i in range(<n_iter>):
            <do something>
            print(prog_mon.get_progress_line(i))
    '''
    def __init__(self, job_size):
        self.start_time = time()
        self.job_size = job_size

    def get_progress_data(self, idx):
        elapsed_time = time() - self.start_time
        progress_rate = (idx + 1) / elapsed_time
        remaining_time_sec = (self.job_size - (idx + 1)) / progress_rate
        prog_percentage = 100.0 * (idx + 1) / self.job_size
        return prog_percentage, remaining_time_sec

    def get_progress_line(self, idx):
        if idx > 0:
            prog_percentage, remaining_time_sec = self.get_progress_data(idx)
            return '%2.0f%% | ~%s remaining' % (prog_percentage, get_sensible_duration(remaining_time_sec))
        else:
            return '0% | TBD'

def get_sensible_duration(duration_sec):
    '''
    Format the given duration as a time string with sec/min/hrs as needed.
    E.g.: get_sensible_duration(500) -> 8.3 min
    '''
    duration_min = duration_sec / 60
    duration_hrs = duration_sec / 3600
    if duration_sec <= 60:
        return '%2.1f sec' % duration_sec
    elif duration_min <= 60:
        return '%2.1f min' % duration_min
    else:
        return '%2.1f hrs' % duration_hrs

def get_latest_filename(target_dir, regexp):
    '''
    Get latest file within a target directory given the regular expression
    '''
    return max(glob.iglob(os.path.join(target_dir, regexp)), key=os.path.getctime)

def setup_logger(log_file, logging_level=logging.INFO):
    ''' Create a logger object to a file '''
    logger = logging.getLogger()
    path, _ = os.path.split(log_file)
    if not os.path.exists(path):
        os.makedirs(path)
    if os.path.isfile(log_file):
        os.remove(log_file)
    hdlr = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(hdlr)
    logger.addHandler(stream_handler)
    logger.setLevel(logging_level)
    return logger

def close_logger(logger):
    ''' Terminate logger '''
    handlers = logger.handlers[:]
    for hdlr in handlers:
        hdlr.close()
        logger.removeHandler(hdlr)


def get_cur_time(format="%Y-%m-%d %H:%M:%S"):
    return strftime(format, localtime())

def get_user_input(title, label):
    ''' Get user input with a window '''
    class window(object):
        def __init__(self, master, label):
            self.master = master
            # top = self.top = Toplevel(master)
            self.l = Label(master, text=label, font='Arial')
            self.l.pack()
            self.e = Entry(master, font='Arial')
            self.e.bind('<Return>', self.cleanup)
            self.e.bind('<Escape>', self.cleanup)
            self.e.pack()
            self.e.focus()

        def cleanup(self, event):
            self.value = self.e.get()
            self.master.destroy()
    root = Tk()
    root.title(title)
    # root.style = Style()
    # root.style.theme_use("alt")
    w = window(root, label)
    root.mainloop()
    return w.value

def timeit(func):
    '''
    Decorator function to time execution of a function.
    Usage:
        <func_output> = timeit(<func>)(<func_args>)
    '''
    def timed(*args, **kwargs):
        t0 = time()
        res = func(*args, **kwargs)
        cprint('%s | %s' % (func.__name__, get_sensible_duration(time() - t0)), 'cyan')
        return res
    return timed

def multi_panel_fig(panels, plot_funcs=None, post_plot_proc=None, titles=None,
                    x_labels=None, y_labels=None, xlims=None, ylims=None, figname=None, is_maximize=None):
    n_row = len(panels)
    n_col = len(panels[0])
    if titles and isinstance(titles[0],(str,)):  # Title only for top panels
        titles = [titles]
    if y_labels and isinstance(y_labels[0],(str,)):  # Labels only for left panels
        y_labels = [[lbl] + [None] * (n_col-1) for lbl in y_labels]
    if x_labels and isinstance(x_labels[0],(str,)):  # Labels only for bottom panels
        x_labels = [[None] * n_col for _ in range(n_row)] + [x_labels]
    fig = plt.figure()
    if figname:
        plt.suptitle(figname, fontsize=16)
    panel_idx = 1
    for i in range(n_row):
        for j in range(n_col):
            try:
                plt.subplot(n_row, n_col, panel_idx)
                if plot_funcs:
                    plot_funcs[i][j](panels[i][j])
                else:
                    plt.imshow(panels[i][j])
                    plt.colorbar()
                if x_labels and x_labels[i][j] is not None:
                    plt.xlabel(x_labels[i][j], fontsize=14)
                if y_labels and y_labels[i][j] is not None:
                    plt.ylabel(y_labels[i][j], fontsize=14)
                if titles and i < len(titles):
                    plt.title(titles[i][j], fontsize=14)
                if xlims:
                    plt.xlim(xlims[i][j])
                if ylims:
                    plt.ylim(ylims[i][j])
                if post_plot_proc:
                    post_plot_proc[i][j]()
            except Exception:
                warn("Error with panel %d:%d" % (i+1,j+1))
            panel_idx += 1

    if is_maximize:
        plt.get_current_fig_manager().full_screen_toggle()
        # mng = plt.get_current_fig_manager()
        # mng.window.wm_geometry("1920x1080+0+0")

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def silentremove(file_or_folder_name):
    if os.path.exists(file_or_folder_name):
        if os.path.isfile(file_or_folder_name):
            os.remove(file_or_folder_name)
        else:  # Folder
            shutil.rmtree(file_or_folder_name)

def overridefolder(folder_path):
    silentremove(folder_path)
    os.makedirs(folder_path)
    return folder_path

def my_basename(s, ext=False):
    basename = os.path.basename(s)
    return basename if ext else os.path.splitext(basename)[0]
                 
def my_print(x, color=None):
    if not color:
        color = 'blue'
    cprint(x, 'blue')
    return time()

def underscore_str(iterable_obj):
    return '_'.join([str(x) for x in iterable_obj])

def starred(s, n_stars=10):
    return '*' * n_stars + f'\n{s}\n' + '*' * n_stars

# def get_available_gpus():
#     local_device_protos = device_lib.list_local_devices()
#     return [int(x.name[-1]) for x in local_device_protos if x.device_type == 'GPU']

def listify(value):
    """ Ensures that the value is a list. If it is not a list, it creates a new list with `value` as an item. """
    # if isinstance(value, Iterable):
    #     return list(value)
    if not isinstance(value, list):
        value = [value]
    return value

def flatten_dict(d):
    def items():
        for key, value in d.items():
            if isinstance(value, dict):
                for subkey, subvalue in flatten_dict(value).items():
                    yield str(key) + "." + str(subkey), subvalue
            else:
                yield key, value

    return dict(items())

def invert_dict(d):
    ''' Including non-unique case '''
    return {v:[k for k in d if d[k] == v] for v in d.values()}

def create_colormap(N):
    return list(itertools.product(np.linspace(0, .7, np.ceil(N ** (1/3))), repeat=3))

class PoolReported():
    def __init__(self, n_threads):
        from multiprocessing.dummy import Pool
        self.pool = Pool(n_threads)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        #Exception handling here
        self.pool.close()

    def map(self, f1, arg_list):
        from tqdm import tqdm
        with tqdm(total=len(arg_list)) as pbar:
            def f2(x):
                y = f1(x)
                pbar.update(1)
                return y
            return self.pool.map(f2, arg_list)

def my_parse(ref_str, s, split_str='_'):
    """
        E.g.,:
            my_parse('timesteps_10', 'timesteps')
    """
    if s not in ref_str: 
        return ''
    ref_str_split = ref_str.split(split_str)
    return ref_str_split[ref_str_split.index(s) + 1]

def easystack(l, stacking_func=np.stack):
    l = [x for x in l if x is not None]
    if len(l) > 0:
        return stacking_func(l)
    else:
        return None

def pickle_save(obj, file, attr='wb'):
    with open(file, attr) as f:
        pickle.dump(obj, f)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

def set_gpu(gpu_list):
    os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join(gpu_list)
    cprint1('(*) CUDA_VISIBLE_DEVICES: {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
    
def get_freer_gpu(utilization=False, tmp_filename=None):
    if tmp_filename is None:
        tmp_filename = platform.node().split('.')[0] + '_tmp'
    if utilization:
        os.system('nvidia-smi -q -d Utilization |grep -A4 GPU|grep Gpu >{}'.format(tmp_filename))
        util = [int(x.split()[2]) for x in open(tmp_filename, 'r').readlines()]
        return np.argmin(util), min(util)
    else:  # By mem
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >{}'.format(tmp_filename))
        memory_available = [int(x.split()[2]) for x in open(tmp_filename, 'r').readlines()]
        return np.argmax(memory_available), max(memory_available)

@contextlib.contextmanager
def dummy_context_mgr():
    yield None

def hist_comparison_fig(dist_dict, bins, **flags):
    for k, v in dist_dict.items():
        assert (isinstance(v, np.ndarray) or
                (isinstance(v, list) and len(v) and isinstance(v[0], np.generic))), \
                    f'Non-ideal input for matplotlib {plt.hist.__name__}: {v}'
        plt.hist(v, bins, alpha=0.5, label=k, **flags)
    plt.legend()
    return plt.gcf()

hist_compare = hist_comparison_fig

def sample_array(a, size=1, axis=0, replace=False):
    a = np.array(a)
    if replace:
        indices = np.array(random.choices(range(a.shape[axis]), k=size))
    else:
        # indices = np.array(random.sample(range(a.shape[axis]), k=size))
        indices = np.random.permutation(a.shape[axis])[:size]
    return np.take(a, indices, axis=axis)

def shuffled(l):
    l = listify(l)
    return [l[i] for i in np.random.permutation(len(l))]

def reshape_sq(a):
    # d = np.sqrt(len(a))
    d = np.sqrt(np.prod(a.shape))
    assert int(d) - d == 0, 'Cannot make array to a square'
    d = int(d)
    return a.reshape(d, d)

def sample_portion(x, q):
    # return random.sample(x, int(len(x) * q))
    # indices = random.sample(range(len(x)), int(len(x) * q))
    # return x[indices]
    return x[np.random.permutation(len(x))[:int(len(x) * q)]]

def chained(l):
    return list(itertools.chain(*l))

def tup2list(tuple_list, tuple_idx):
    return list(zip(*tuple_list))[tuple_idx]

def unique_keeporder(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]
