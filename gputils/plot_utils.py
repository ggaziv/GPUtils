"""
    General useful utility functions for plotting/visualization
"""


import warnings
import gputils.startup_guyga as gputils
import seaborn as sns
import numpy as np 
import itertools
from collections import namedtuple
import pandas as pd
from collections import deque


def set_axes_equal(ax):
    """Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc..  

    This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.
    ref: https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to

    Args:
        ax: a matplotlib axis, e.g., as output from plt.gca().
    """  
    
    limits_dict = {ax_name: getattr(ax, f'get_{ax_name}lim3d')() for ax_name in ['x', 'y', 'z']}
    ranges_dict = {ax_name: abs(np.diff(limits)) for ax_name, limits in limits_dict.items()}
    middle_dict = {ax_name: np.mean(limits) for ax_name, limits in limits_dict.items()}
    
    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max(list(ranges_dict.values()))
    
    for ax_name, middle in middle_dict.items():
        getattr(ax, f'set_{ax_name}lim3d')([int(middle - plot_radius), int(middle + plot_radius)])
    
    
def add_unity_ref_line(ax=None, inverse=False, alpha=.3):
    if ax is None:
        ax = gputils.plt.gca()
    # lim = min(ax.get_xlim()[-1], ax.get_ylim()[-1])
    # ax.plot((0, lim), (0, lim), '-k', alpha=.3)
    lim1 = max(ax.get_xlim()[0], ax.get_ylim()[0])
    lim2 = min(ax.get_xlim()[-1], ax.get_ylim()[-1])
    ax.plot((lim1, lim2), ((1 - 2 * inverse) * lim1, (1 - 2 * inverse) * lim2), '-k', alpha=alpha)
    
    
def add_unity_ref_planes(ax, alpha=0.2):
    limits_dict = {ax_name: getattr(ax, f'get_{ax_name}lim3d')() for ax_name in ['x', 'y', 'z']}
    lim_max = max(lim[1] for lim in limits_dict.values())
    lim_min = max(lim[0] for lim in limits_dict.values())
    aa, bb = np.meshgrid(*([np.array((lim_min, lim_max))] * 2))  
    deq = deque(['x', 'y', 'z'])
    d = dict(zip(['x', 'y', 'z'], [aa, bb, aa]))
    for _ in range(len(deq)):
        # ax_names_other, ax_name = (deq[0], deq[1]), deq[2]
        # ax_names_other = [ax_name1 for ax_name1 in limits_dict if ax_name1 != ax_name]
        # aa, bb = np.meshgrid(*[array(limits_dict[ax_name1]) for ax_name1 in ax_names_other])
        # aa, bb = np.meshgrid(*([array((-100, 100))] * 2))
        # d = dict(zip((ax_names_other + [ax_name]), [aa, bb, (aa + bb)/2]))
        # d = dict(zip((ax_names_other + [ax_name]), [aa, bb, aa]))
        # print(deq)
        # ax.plot_surface(*[d[ax_name1] for ax_name1 in ['x', 'y', 'z']], alpha=alpha)
        ax.plot_surface(*[d[ax_name1] for ax_name1 in deq], alpha=alpha)
        deq.rotate(1)
        

class ErrorBarred():
    def __init__(self, plotter=sns.scatterplot):
        self.plotter = plotter
        
    def __call__(self, data: pd.DataFrame, x: str, y: str, hue: str=None, 
                 boot_var: str=None, estimator='mean', 
                 errorbar=('ci', 95), n_boot=100, 
                 elinewidth=2, ecolor='k',capsize=2,
                 err_alpha=None, errorevery=1, seed=None, 
                 n_threads=None, **kwargs):
        
        agg = sns._statistics.EstimateAggregator(estimator, errorbar, n_boot=n_boot, seed=seed)
        cols_exclude = [x, y, boot_var]
        columns_group = [col_name for col_name in data.columns if col_name not in cols_exclude]
        # data = data.dropna(subset=[x, y])
        isna = data.isna()
        data = data[~isna[x] & ~isna[y]]
        # it = itertools.product(*[data[k].unique() for k in columns_group])
        it = data[columns_group].drop_duplicates().itertuples(index=False, name=None)
        groupped = data.groupby(columns_group)
        value_tup = namedtuple('Value', ['val', 'err_min', 'err_max'])
        def extract_res(g, var): 
            df_agg = groupped.get_group(g)
            if len(df_agg) == 1:
                return value_tup(float(df_agg[var]), 0, 0)
            res = agg(df_agg, var)
            return value_tup(res[f"{var}"], res[f"{var}"]-res[f"{var}min"], res[f"{var}max"]-res[f"{var}"])
            # except:
            #     warnings.warn(f"Group {g} not found. Skipping")
            #     return value_tup(0, 0, 0)
                
        point = namedtuple('Point', ['x', 'y'])
        if n_threads is None:
            err_list = []
            for g in it:
                err_list.append(point(*[extract_res(g, var) for var in [x, y]]))
        else:
            with gputils.Pool(n_threads) as pool:
                err_list = pool.map(lambda g: point(*[extract_res(g, var) for var in [x, y]]), list(it))
        
        data1 = data.groupby(columns_group).mean().reset_index()
        g = self.plotter(data=data1, x=x, y=y, hue=hue, **kwargs)
        x_vals, y_vals, x_errs_min, x_errs_max, y_errs_min, y_errs_max = list(zip(*[(err.x.val, err.y.val, 
                                                                                     err.x.err_min, err.x.err_max, 
                                                                                     err.y.err_min, err.y.err_max) for err in err_list]))
        g.errorbar(x_vals, 
                   y_vals, 
                   xerr=[x_errs_min, x_errs_max], 
                   yerr=[y_errs_min, y_errs_max],
                   fmt=' ',
                   elinewidth=elinewidth,
                   capsize=capsize,
                   ecolor=ecolor,
                   alpha=err_alpha,
                   errorevery=errorevery)
        return g
                             