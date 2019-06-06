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
__credits__ = ["Guy Gaziv"]
__email__ = "guy.gaziv@weizmann.ac.il"

# print('>> Running startup script.')
from pprint import pprint
import sys, os
from os.path import join as pjoin
from datetime import datetime
from numpy import stack, vstack, dstack, hstack, array, load, inf, squeeze, mean, median, linspace, prod
from GPUtils.utils1 import *
import random, time
from numpy.random import randint, randn, random_sample
from termcolor import cprint
cprint1 = lambda s: cprint(s, 'cyan', attrs=['bold'])
from sys import getsizeof
from scipy.io import loadmat; loadmat = report(loadmat, 'Loading')
from scipy.io import savemat; savemat = report(savemat, 'Saving')
import scipy.misc as smisc
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 14})
# load = report(load, 'Loading')
from multiprocessing.dummy import Pool
from tqdm import tqdm
from collections import OrderedDict
from natsort import natsorted
from importlib import reload
import shutil, pickle
import itertools
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import h5py
warnings.resetwarnings()
warnings.simplefilter("ignore", ResourceWarning)
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.filterwarnings("ignore",category=DeprecationWarning)
