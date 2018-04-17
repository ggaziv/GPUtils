#!/usr/bin/env python
"""
    General useful utility functions
    Date created: 17/4/18
    Python Version: 3.5
"""

import os
from tkinter import *
from time import localtime, strftime, time
from termcolor import cprint
import logging
import glob
import numpy as np
from matplotlib import pyplot as plt
from warnings import warn

__author__ = "Guy Gaziv"
__credits__ = ["Guy Gaziv"]
__email__ = "guy.gaziv@weizmann.ac.il"

def report(func, msg):
    '''
    Decorator function to print a message before execution of a function and report its execution time
    Usage:
        <func_output> = reported(<func>, <msg>)(<func_args>)
    '''
    def reported(*args, **kwargs):
        cprint(msg, 'magenta')
        t1 = time()
        res = func(*args, **kwargs)
        cprint('Done in %s.' % get_sensible_duration(time()-t1), 'magenta')
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
        prog_percentage, remaining_time_sec = self.get_progress_data(idx)
        return '%2.0f%% | ~%s remaining' % (prog_percentage, get_sensible_duration(remaining_time_sec))

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
                if x_labels:
                    plt.xlabel(x_labels[i][j])
                if y_labels:
                    plt.ylabel(y_labels[i][j])
                if titles:
                    plt.title(titles[i][j])
                if xlims:
                    plt.xlim(xlims[i][j])
                if ylims:
                    plt.ylim(ylims[i][j])
                if post_plot_proc:
                    post_plot_proc[i][j]()
            except Exception:
                warn("Error with panel %d:%d" % (i,j))
            panel_idx += 1

    if is_maximize:
        plt.get_current_fig_manager().full_screen_toggle()
        # mng = plt.get_current_fig_manager()
        # mng.window.wm_geometry("1920x1080+0+0")
