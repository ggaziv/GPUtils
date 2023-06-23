"""
    General useful utility functions.
    To run at any startup of IPython:
        setenv PYTHONSTARTUP <path_to_this_script>
    Similarly for non-interactive Python:
        from <...>startupscript_guyga import *

    Date created: 17/4/18
    Python Version: 3.5
"""

__author__ = "Guy Gaziv"

# print('>> Running startup script.')
from pprint import pprint
import sys, os, copy, math, json, platform
# from dotmap import DotMap
from os.path import join as pjoin, exists as pexists, basename, isfile, dirname
from datetime import datetime
from numpy import stack, vstack, dstack, hstack, array, load, save, inf, squeeze, mean, median, std, linspace, prod, arange, \
    nanmean, nanmedian, nanstd, nanmax, nanmin, round, zeros, ones, where, unique, reshape, sqrt, isnan, ma, moveaxis, \
    percentile
import pandas as pd, xarray as xr
from gputils.utils1 import *
import random, time
from numpy.random import randint, randn, random_sample
from termcolor import cprint
cprint1 = lambda s, *args, **kwargs: cprint(s, 'cyan', attrs=['bold'], *args, **kwargs)
cprintrb = lambda s, *args, **kwargs: cprint(s, 'red', attrs=['bold'], *args, **kwargs)
cprintkb = lambda s, *args, **kwargs: cprint(s, attrs=['bold'], *args, **kwargs)
cprintbb = lambda s, *args, **kwargs: cprint(s, 'blue', attrs=['bold'], *args, **kwargs)
cprintc = lambda s, *args, **kwargs: cprint(s, 'cyan', *args, **kwargs)
cprintm = lambda s, *args, **kwargs: cprint(s, 'magenta', *args, **kwargs)
from sys import getsizeof
import scipy
from scipy import io as sio
from scipy import signal
from scipy.io import loadmat; loadmat = report(loadmat, 'Loading')
from scipy.io import savemat; savemat = report(savemat, 'Saving')
import scipy.misc as smisc
from scipy import stats
from scipy.spatial import distance
from matplotlib import pyplot as plt
import seaborn as sns
plt.rcParams.update({'font.size': 14})
# load = report(load, 'Loading')
from multiprocessing.dummy import Pool
from tqdm import tqdm
from collections import OrderedDict
from natsort import natsorted, index_natsorted
from importlib import reload
import shutil, pickle
import itertools, functools
from itertools import chain, permutations, combinations, product
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import h5py
from PIL import Image
# import nibabel as nib
import zipfile
import argparse
from argparse import Namespace
from dataclasses import dataclass, field
import pyrallis

warnings.resetwarnings()
warnings.simplefilter("ignore", ResourceWarning)
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.filterwarnings("ignore",category=DeprecationWarning)
