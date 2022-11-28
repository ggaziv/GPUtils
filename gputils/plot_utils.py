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
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)


def set_pub():
    gputils.plt.rcParams.update({
        "font.weight": "bold",  # bold fonts
        "tick.labelsize": 15,   # large tick labels
        "lines.linewidth": 1,   # thick lines
        "lines.color": "k",     # black lines
        "grid.color": "0.5",    # gray gridlines
        "grid.linestyle": "-",  # solid gridlines
        "grid.linewidth": 0.5,  # thin gridlines
        "savefig.dpi": 300,     # higher resolution output.
    })
    
    
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
                 post_agg_fn_x=None, post_agg_fn_y=None,
                 elinewidth=2, ecolor='k',capsize=2,
                 err_alpha=None, errorevery=1, seed=None, 
                 n_threads=None, palette=None, **kwargs):
        if palette is None:
            palette = sns.color_palette('bright', len(data.reset_index()[hue].unique()))
        
        cols_exclude = [x, y, boot_var]
        columns_group = [col_name for col_name in data.columns if col_name not in cols_exclude]
        data1 = data.groupby(columns_group).mean().reset_index().copy()
        for col_name, post_agg_fn in zip((x,y), (post_agg_fn_x, post_agg_fn_y)):
            if post_agg_fn is not None:
                data1[col_name] = post_agg_fn(data1[col_name])
        g = self.plotter(data=data1, x=x, y=y, hue=hue, palette=palette, **kwargs)
        
        if boot_var is not None:
            err_list = self.compute_errorbars(data, x, y, columns_group, 
                                              estimator, errorbar, n_boot, 
                                              post_agg_fn_x, post_agg_fn_y, 
                                              seed)
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
    
    def compute_errorbars(self, data, x, y, columns_group, 
                          estimator, errorbar, n_boot, 
                          post_agg_fn_x, post_agg_fn_y, 
                          seed):
        agg = sns._statistics.EstimateAggregator(estimator, errorbar, n_boot=n_boot, seed=seed)
        # data = data.dropna(subset=[x, y])
        isna = data.isna()
        data = data[~isna[x] | ~isna[y]]
        # it = itertools.product(*[data[k].unique() for k in columns_group])
        it = data[columns_group].drop_duplicates().itertuples(index=False, name=None)
        groupped = data.groupby(columns_group)
        value_tup = namedtuple('Value', ['val', 'err_min', 'err_max'])
        def extract_res(g, var, post_agg_fn): 
            df_agg = groupped.get_group(g)
            df_agg.dropna(subset=var)
            if len(df_agg) == 0:
                return None
            elif len(df_agg) == 1:
                res = df_agg[var].iloc[0]
                if post_agg_fn is not None:
                    res = post_agg_fn(res)
                return value_tup(res, 0, 0)
            else:
                res = agg(df_agg, var)
                if post_agg_fn is not None:
                    res[f"{var}"], res[f"{var}min"], res[f"{var}max"] = map(post_agg_fn, (res[f"{var}"], res[f"{var}min"], res[f"{var}max"]))
                return value_tup(res[f"{var}"], res[f"{var}"]-res[f"{var}min"], res[f"{var}max"]-res[f"{var}"])
            # except:
            #     warnings.warn(f"Group {g} not found. Skipping")
            #     return value_tup(0, 0, 0)     
        point = namedtuple('Point', ['x', 'y'])
        # if n_threads is None:
        err_list = []
        for g in it:
            point_xy = [extract_res(g, var, post_agg_fn) for var, post_agg_fn in zip([x, y], (post_agg_fn_x, post_agg_fn_y))]
            if not None in point_xy:
                err_list.append(point(*point_xy))
            # err_list.append(point(*[extract_res(g, var) for var in [x, y]]))
        # else:
        #     with gputils.Pool(n_threads) as pool:
        #         err_list = pool.map(lambda g: point(*[extract_res(g, var) for var in [x, y]]), list(it))
        
        return err_list
                                    

class AnnotatedScatter():
    """Add tooltip images on hover on scatterplot
    """
    def __init__(self, ax, scatter_list, fig=None):
        self.ax = ax
        self.scatter_list = scatter_list
        if fig is None:
            self.fig = gputils.plt.gcf()
        else:
            self.fig = fig
        self.ab = AnnotationBbox(OffsetImage(np.zeros((100,100,3)), zoom=0.2), 
                            (0,0),
                            # xybox=(0,0),
                            xybox=(60., -60.),
                            xycoords='data',
                            boxcoords="offset points",
                            # pad=0.5,
                            bboxprops=dict(linewidth=0.),
                            # arrowprops=dict(
                            #     # arrowstyle="->",
                            #     # connectionstyle="angle,angleA=0,angleB=90,rad=3"
                            # )
                            )
        self.ax.add_artist(self.ab)
        self.ab.set_visible(False)

    def hover(self, event):
        # ab.set_visible(True)
        vis = self.ab.get_visible()
        if event.inaxes == self.ax:
            for sc, sc_images in self.scatter_list:
                is_contained, items = sc.contains(event)
                if is_contained:
                    point_index = items['ind'][0]
                    pos = sc.get_offsets()[point_index]
                    self.ab.xy = pos
                    self.ab.offsetbox = OffsetImage(np.array(sc_images[point_index]), zoom=0.4, zorder=10)
                    self.ab.set_visible(True)
                    self.fig.canvas.draw_idle()
                    return
            if vis:
                self.ab.set_visible(False)
                self.fig.canvas.draw_idle()
                