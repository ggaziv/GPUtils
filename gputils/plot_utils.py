"""
    General useful utility functions for plotting/visualization
"""


import gputils.startup_guyga as gputils
import numpy as np 


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
    for ax_name in limits_dict:
        ax_names_other = [ax_name1 for ax_name1 in limits_dict if ax_name1 != ax_name]
        # aa, bb = np.meshgrid(*[array(limits_dict[ax_name1]) for ax_name1 in ax_names_other])
        # aa, bb = np.meshgrid(*([array((-100, 100))] * 2))
        aa, bb = np.meshgrid(*([np.array((lim_min, lim_max))] * 2))
        d = dict(zip((ax_names_other + [ax_name]), [aa, bb, (aa + bb)/2]))
        ax.plot_surface(*[d[ax_name1] for ax_name1 in ['x', 'y', 'z']], alpha=alpha)