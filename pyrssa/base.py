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


# This line solves problem for PyCharm: module 'backend_interagg' has no attribute 'FigureCanvas'...
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
def data(name):
    """
    Function for loading available in pyrssa package datasets. Available datasets are stored in the data directory.

    :param name: Name of dataset to load
    :return:
    """
    return read_csv(os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data"), f'{name}.csv'))


def parestimate(x, groups, method="esprit", subspace="column", normalize_roots=None, dimensions=None,
                solve_method="ls", drop=True):
    """
    Function to estimate the parameters (frequencies and rates) given a set of SSA eigenvectors.

    :param x: SSA object
    :param groups: list of indices of eigenvectors to estimate from
    :param method: For 1D-SSA,
        Toeplitz SSA, and MSSA: parameter estimation method, 'esprit' for 1D-ESPRIT (Algorithm 3.3 in Golyandina et al (
        2018)), 'pairs' for rough estimation based on pair of eigenvectors (Algorithm 3.4 in Golyandina et al (2018)).
        For nD-SSA: parameter estimation method. For now only 'esprit' is supported (Algorithm 5.6 in Golyandina et al (
        2018)). lowest dimension, when possible (length of groups is one)
    :param subspace: which subspace will be used for parameter estimation
    :param normalize_roots: logical vector or None, force signal roots to lie on unit circle.
        None means automatic selection: normalize iff circular topology OR Toeplitz SSA used
    :param dimensions: a vector of dimension indices to perform ESPRIT along. None means all dimensions.
    :param solve_method: approximate matrix equation solving method, 'ls' for least-squares,
        'tls' for total-least-squares.
    :param drop: logical, if 'TRUE' then the result is coerced to lowest dimension,
        when possible (length of groups is one)

    :return:

    """
    return Parestimate(x=x, groups=groups, method=method, subspace=subspace, normalize_roots=normalize_roots,
                       dimensions=dimensions, solve_method=solve_method, drop=drop)


def ssa(x, L=None, neig=None, mask=None, wmask=None, kind="1d-ssa", circular=None,
        column_projector="none", row_projector="none", svd_method="auto"):
    """
    Description
    -------

    Set up the SSA object and perform the decomposition, if necessary.

    :param x: object to be decomposed.
    :type x: Union[class:`pd.DataFrame`, class:`pd.Series`]
    :param L: window length. Fixed to half of the series length by default.
        Should be vector of length 2 for 2d SSA
    :type L: int, optional
    :param neig: number of desired eigentriples. If None, then sane default value
        will be used.
    :type neig: int, optional
    :param mask: for shaped 2d SSA case only. Logical matrix with same dimension as x.
        Specifies form of decomposed array. If None, then all non-NA elements will be used
    :param wmask: for shaped 2d SSA case only. Logical matrix which specifies window form.
    :param kind: SSA method. This includes ordinary 1d SSA, 2d SSA, Toeplitz variant of 1d SSA, multichannel
        variant of SSA and complex SSA. Defaults to 1d SSA.
    :type kind: str, optional
    :param circular: logical vector of one or two elements, describes series topology for 1d SSA and Toeplitz SSA
        or field topology for 2d SSA. 'TRUE' means series circularity for 1d case or circularity
        by a corresponding coordinate for 2d case. See Shlemov and Golyandina (2014) for more information.
    :param column_projector, row_projector: column and row signal subspaces projectors for SSA with projection.
    :type column_projector: int, optional
    :type row_projector: int, optional
    :param svd_method: 	singular value decomposition method.
    :return: SSA Object. The precise layout of the object is mostly meant opaque and subject to
        change in different version of the package.
    :rtype: class:`SSA`

    Details
    -------

    This is the main entry point to the package. This routine constructs the SSA object filling all necessary
    internal structures and performing the decomposition if necessary. For the comprehensive description of SSA
    modifications and their algorithms see Golyandina et al (2018).

    Variants of SSA


    Some details for this SSA.

    """
    return SSA(x, L=L, neig=neig, mask=mask, wmask=wmask, kind=kind,
               column_projector=column_projector,
               row_projector=row_projector,
               svd_method=svd_method,
               call=_get_call(inspect.currentframe().f_back))


def reconstruct(ds, groups):
    return Reconstruction(ds, groups)


def iossa(x: SSA, nested_groups, tol=1e-5, kappa=2, maxiter=100, norm=None, trace=False, kappa_balance=0.5, **kwargs):
    return IOSSA(x=x, nested_groups=nested_groups, tol=tol, kappa=kappa, maxiter=maxiter, norm=norm, trace=trace,
                 kappa_balance=kappa_balance, call=_get_call(inspect.currentframe().f_back), **kwargs)


# Weighted correlations
def wcor(ds, groups=range(1, 50)):
    return WCorMatrix(ds, groups)


# Forecasting functions

def rforecast(ds, groups, length=1, base="reconstructed", only_new=True, reverse=False,
              drop=False, drop_attributes=False, cache=True, **kwargs):
    return RForecast(ds, groups, length=length, base=base, only_new=only_new, reverse=reverse,
                     drop=drop, drop_attributes=drop_attributes, cache=cache, **kwargs)


def vforecast(ds, groups, length=1, only_new=True, drop=False, drop_attributes=False, **kwargs):
    return VForecast(ds, groups, length=length, only_new=only_new, drop=drop, drop_attributes=drop_attributes, **kwargs)


def bforecast(ds, groups, length=1, R=100, level=0.95, kind="recurrent", interval="confidence",
              only_new=True, only_intervals=False, drop=True, drop_attributes=False, cache=True, **kwargs):
    return BForecast(ds, groups, length=length, r=R, level=level, kind=kind, interval=interval, only_new=only_new,
                     only_intervals=only_intervals, drop=drop, drop_attributes=drop_attributes, cache=cache, **kwargs)


# Plot functions

def _vector_plot(dt: SSA, idx, contrib=True, layout=None):
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


def _paired_plot(dt: SSA, idx, contrib=True):
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


def _should_share_limits(series_arr, max_diff=1):
    range_values = [[min(ser), max(ser)] for ser in series_arr]
    arr_start, arr_end = list(map(list, zip(*range_values)))
    min_start = min(arr_start)
    max_end = max(arr_end)
    max_start = max(arr_start)
    min_end = min(arr_end)
    max_range = abs(min_start - max_end)
    return abs(max_start - min_start) / max_range <= max_diff and abs(max_end - min_end) / max_range <= max_diff


def _xyplot(dt: Reconstruction, x, add_residuals, add_original, layout, superpose):

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
            share_y = _should_share_limits(all_series)
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


def _matplot(dt: Reconstruction, x, add_residuals, add_original):

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


def _sigma_plot(ts: SSA):
    plt.suptitle('Component norms')
    plt.plot(ts.sigma, marker='o')
    plt.xlabel('Index')
    plt.ylabel('Norms')
    plt.yscale('log')
    plt.show()


def _wcor_plot(wcor_matrix: WCorMatrix, scales=None):
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
        return _vector_plot(ts, idx, contrib=contrib)
    elif kind == "paired":
        return _paired_plot(ts, idx, contrib=contrib)
    elif type(ts) == Reconstruction:
        if method == "matplot":
            return _matplot(ts, x, add_residuals, add_original)
        if method == "xyplot":
            return _xyplot(ts, x, add_residuals, add_original, layout, superpose)
    elif type(ts) == SSA:
        return _sigma_plot(ts)
    elif type(ts) == WCorMatrix:
        return _wcor_plot(ts, scales)
