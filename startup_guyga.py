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

from __future__ import print_function
# print('>> Running startup script.')
from pprint import pprint
import sys, os, time
from numpy import *
from utils.general_purpose_utils import *
from termcolor import cprint
cprint1 = lambda s: cprint(s, 'cyan', attrs=['bold'])
from sys import getsizeof
from scipy.io import loadmat; loadmat = report(loadmat, 'Loading')
from matplotlib import pyplot as plt
load = report(load, 'Loading')
from multiprocessing.dummy import Pool
from tqdm import tqdm
from collections import OrderedDict
from natsort import natsorted
import shutil, pickle
