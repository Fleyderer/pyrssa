from pyrssa import SSA, IOSSA, Parestimate
from pyrssa import Reconstruction
from pyrssa import RForecast, VForecast, BForecast
from pyrssa import WCorMatrix
from pyrssa import installer
from pyrssa.conversion import pyrssa_conversion_rules
from rpy2 import robjects
import rpy2.robjects.conversion as conversion
import rpy2.robjects.packages as rpackages
from rpy2.rinterface_lib import callbacks
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pandas import read_csv
import numpy as np
import os
import inspect
from typing import Callable


# This line solve problem for PyCharm: module 'backend_interagg' has no attribute 'FigureCanvas'...
matplotlib.use('TkAgg')

# Set conversion rules
conversion.set_conversion(pyrssa_conversion_rules)

# Ignore warnings
callbacks.consolewrite_warnerror = lambda *args: None

# Install required R packages.
installer.install_required()

r = robjects.r
r_ssa = rpackages.importr('Rssa')


def _get_call(frame):
    call = inspect.getframeinfo(frame)[3][0]
    return call[call.find('=') + 1:].strip().rstrip()


# Read pyrssa dataframes
def data(ds_name):
    return read_csv(os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data"), f'{ds_name}.csv'))


def parestimate(x, groups, method="esprit", subspace="column", normalize_roots=None, dimensions=None,
                solve_method="ls", drop=True):
    return Parestimate(x=x, groups=groups, method=method, subspace=subspace, normalize_roots=normalize_roots,
                       dimensions=dimensions, solve_method=solve_method, drop=drop)


def ssa(ds, L=None, kind="1d-ssa", column_projector="none", row_projector="none", svd_method="auto"):
    return SSA(ds, L=L, kind=kind,
               column_projector=column_projector,
               row_projector=row_projector,
               svd_method=svd_method,
               call=_get_call(inspect.currentframe().f_back))


def iossa(x: SSA, nested_groups, tol=1e-5, kappa=2, maxiter=100, norm=None, trace=False, kappa_balance=0.5, **kwargs):
    return IOSSA(x=x, nested_groups=nested_groups, tol=tol, kappa=kappa, maxiter=maxiter, norm=norm, trace=trace,
                 kappa_balance=kappa_balance, call=_get_call(inspect.currentframe().f_back), **kwargs)


def reconstruct(ds, groups):
    return Reconstruction(ds, groups)


def wcor(ds, groups=range(1, 50)):
    return WCorMatrix(ds, groups)


def rforecast(ds, groups, length=1, base="reconstructed", only_new=True,
              drop=False, drop_attributes=False, cache=True, **kwargs):
    return RForecast(ds, groups, length=length, base=base, only_new=only_new,
                     drop=drop, drop_attributes=drop_attributes, cache=cache, **kwargs)


def vforecast(ds, groups, length=1, only_new=True, drop=False, drop_attributes=False, **kwargs):
    return VForecast(ds, groups, length=length, only_new=only_new, drop=drop, drop_attributes=drop_attributes, **kwargs)


def bforecast(ds, groups, length=1, R=100, level=0.95, kind="recurrent", interval="confidence",
              only_new=True, only_intervals=False, drop=True, drop_attributes=False, cache=True, **kwargs):
    return BForecast(ds, groups, length=length, r=R, level=level, kind=kind, interval=interval, only_new=only_new,
                     only_intervals=only_intervals, drop=drop, drop_attributes=drop_attributes, cache=cache, **kwargs)


# Plot functions


# Deprecated
def vector_plot_old(dt, idx):
    if idx is None:
        idx = range(len(dt.U))
    cols = 4
    rows = round(len(idx) / cols)
    gs1 = gridspec.GridSpec(rows, cols)
    gs1.update(wspace=0, hspace=0)
    fig, ax = plt.subplots(rows, cols)
    plt.setp(ax.flat, adjustable='box')
    fig.suptitle("Eigenvectors")
    eig_n = 0
    for row in range(rows):
        for col in range(cols):
            ax[row, col].plot(range(len(dt.U[idx[eig_n]])), dt.U[idx[eig_n]], label=eig_n)

            ax[row, col].set_xticks([])
            ax[row, col].set_yticks([])
            ax[row, col].legend()
            eig_n += 1
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def vector_plot(dt: SSA, idx, contrib=True, layout=None):
    if idx is None:
        idx = range(1, min(10, len(dt.U)) + 1)
    if contrib is True:
        cntrb = dt.contributions(idx)
    else:
        cntrb = None

    if layout is None:
        cols = 4
        rows = int(np.ceil(len(idx) / cols))
    else:
        cols = layout[0]
        rows = layout[1]

    fig = plt.figure(figsize=(cols + 1, rows + 1))
    fig.tight_layout(h_pad=1)
    gs = gridspec.GridSpec(rows, cols, width_ratios=[1, 1, 1, 1], figure=fig, wspace=0, hspace=0.5)
    fig.suptitle("Eigenvectors")
    ax = None
    for i in range(len(idx)):
        ax = fig.add_subplot(gs[i], sharey=ax)
        ax.plot(range(len(dt.U[idx[i] - 1])), dt.U[idx[i] - 1])
        if cntrb is None:
            ax.set_title(idx[i])
        else:
            ax.set_title(f'{idx[i]} ({cntrb[i] * 100:.2f} %)')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect(aspect='auto', adjustable='box')
    plt.show()


def paired_plot(dt: SSA, idx, contrib=True):
    if idx is None:
        idx = range(len(dt.U))
    if contrib is True:
        cntrb = dt.contributions(idx)
    else:
        cntrb = None
    cols = 5
    rows = int(np.ceil((len(idx) - 1) / cols))
    fig = plt.figure(figsize=(cols + 1, rows + 1))
    fig.tight_layout(h_pad=1)
    gs = gridspec.GridSpec(rows, cols, width_ratios=[1] * cols, figure=fig, wspace=0, hspace=0.5)

    fig.suptitle("Pairs of eigenvectors")
    for i in range(len(idx) - 1):
        ax = fig.add_subplot(gs[i])
        ax.plot(dt.U[idx[i + 1] - 1], dt.U[idx[i] - 1])
        if cntrb is None:
            ax.set_title(f'{idx[i]} vs {idx[i + 1]}')
        else:
            ax.set_title(f'{idx[i]} ({cntrb[i] * 100:.2f}%) vs {idx[i + 1]} ({cntrb[i + 1] * 100:.2f}%)')
        ax.set_xticks([])
        ax.set_yticks([])
        ratio = 1.0
        xleft, xright = ax.get_xlim()
        ybottom, ytop = ax.get_ylim()
        ax.set_aspect(abs((xright - xleft) / (ybottom - ytop)) * ratio)
    plt.show()


def should_share_limits(series_arr, max_diff=1):
    range_values = [[min(ser), max(ser)] for ser in series_arr]
    arr_start, arr_end = list(map(list, zip(*range_values)))
    min_start = min(arr_start)
    max_end = max(arr_end)
    max_start = max(arr_start)
    min_end = min(arr_end)
    max_range = abs(min_start - max_end)
    return abs(max_start - min_start) / max_range <= max_diff and abs(max_end - min_end) / max_range <= max_diff


def xyplot(dt: Reconstruction, x, add_residuals, add_original, layout, superpose):

    if x is None:
        x = dt.series.index

    if superpose:
        fig, ax = plt.subplots()
        fig.tight_layout()
        if add_original:
            ax.plot(x, dt.series, label="Original")
        for name in dt.names:
            ax.plot(x, getattr(dt, name), label=name)
        if add_residuals:
            ax.plot(x, dt.residuals, label='Residuals')
        ax.legend()
        plt.title(label="Reconstructed series")
        plt.show()

    else:

        cnt = len(dt.names) + add_original + add_residuals

        if layout is not None:
            if layout[0] * layout[1] < cnt:
                raise ValueError(f"Layout size {layout} is less than count of plotting series ({cnt}).")
            rows = layout[0]
            cols = layout[1]
        else:
            rows = cnt
            cols = 1

        plotting_series = []
        if add_original:
            plotting_series.append({'name': 'Original', 'series': dt.series})
        for i in range(add_original, cnt - add_residuals):
            name = dt.names[i - add_original]
            plotting_series.append({'name': name, 'series': getattr(dt, name)})
        if add_residuals:
            plotting_series.append({'name': 'Residuals', 'series': dt.residuals})

        if cols > 1:
            all_series = [ser['series'] for ser in plotting_series]
            share_y = should_share_limits(all_series)
        else:
            share_y = False

        fig = plt.figure(figsize=(2 * (cols + 1), 2 * (rows + 1)))
        if share_y:
            gs = gridspec.GridSpec(rows, cols, width_ratios=[1] * cols, figure=fig, wspace=0.1, hspace=0.5)
        else:
            gs = gridspec.GridSpec(rows, cols)
        fig.suptitle("Reconstructed series")
        ax = None

        for i in range(len(plotting_series)):
            if share_y:
                is_first = ax is None
                ax = fig.add_subplot(gs[i], sharex=ax, sharey=ax)
                if not is_first:
                    ax.tick_params(left=False, labelleft=False)
            else:
                ax = fig.add_subplot(gs[i], sharex=ax)
            ax.plot(x, plotting_series[i]['series'])
            ax.set_title(plotting_series[i]['name'])

        if not share_y:
            fig.tight_layout()
        plt.show()


def matplot(dt: Reconstruction, x, add_residuals, add_original):

    if x is None:
        x = dt.series.index

    fig, ax = plt.subplots()
    if add_original:
        ax.plot(x, dt.series, label="Original", color="black", linewidth=0.5)
    for name in dt.names:
        ax.plot(x, getattr(dt, name), label=name, linestyle="dashed")
    if add_residuals:
        ax.plot(x, dt.residuals, label='Residuals', linestyle="dashed")
    ax.legend()
    plt.title(label="Reconstructed series")
    plt.show()


def sigma_plot(ts: SSA):
    plt.suptitle('Component norms')
    plt.plot(ts.sigma, marker='o')
    plt.xlabel('Index')
    plt.ylabel('Norms')
    plt.yscale('log')
    plt.show()


def wcor_plot(wcor_matrix: WCorMatrix, scales=None):
    plt.imshow(np.asarray(wcor_matrix.matrix), cmap='gray_r', vmin=0, vmax=1)
    plt.gca().invert_yaxis()

    if scales is None:
        ticks = range(len(wcor_matrix.groups))
        labels = wcor_matrix.groups
    else:
        ticks = np.array(scales) - 1  # fix for indexing of components on the plot
        labels = scales

    plt.title("W-correlation matrix")
    plt.xticks(ticks, labels=labels)
    plt.yticks(ticks, labels=labels)
    plt.show()


def plot(ts, x=None, kind=None, add_residuals=True, add_original=True, idx=None,
         scales=None, contrib=True, layout=None, superpose=False, method=None):
    if kind == "vectors":
        return vector_plot(ts, idx, contrib=contrib)
    elif kind == "paired":
        return paired_plot(ts, idx, contrib=contrib)
    elif type(ts) == Reconstruction:
        if method == "matplot":
            return matplot(ts, x, add_residuals, add_original)
        if method == "xyplot":
            return xyplot(ts, x, add_residuals, add_original, layout, superpose)
    elif type(ts) == SSA:
        return sigma_plot(ts)
    elif type(ts) == WCorMatrix:
        return wcor_plot(ts, scales)
