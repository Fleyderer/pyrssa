from pyrssa import SSA
from pyrssa import Reconstruction
from pyrssa import RForecast, VForecast, BForecast
from pyrssa import WCorMatrix
from pyrssa import installer
from rpy2 import robjects
import rpy2.robjects.packages as rpackages
import rpy2.robjects.conversion as conversion
from rpy2.robjects import default_converter
from rpy2.robjects.numpy2ri import converter as numpy_converter
from rpy2.robjects.pandas2ri import converter as pandas_converter
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pandas import read_csv
import numpy as np

import os

# Install required R packages.
installer.install_required()

r = robjects.r
r_ssa = rpackages.importr('Rssa')


class FloatVector(robjects.FloatVector):
    def __init__(self, obj):
        super().__init__(obj)

    def __mul__(self, other):
        if isinstance(other, float):
            return r.sapply(self, "*", other)
        elif isinstance(other, int):
            return r.sapply(self, "*", other)

    def __truediv__(self, other):
        if isinstance(other, float):
            return r.sapply(self, "/", other)
        elif isinstance(other, int):
            return r.sapply(self, "/", other)

    def __add__(self, other):
        if isinstance(other, float):
            return r.sapply(self, "+", other)
        elif isinstance(other, int):
            return r.sapply(self, "+", other)

    __rmul__ = __mul__

    __radd__ = __add__


class IntVector(robjects.IntVector):
    def __init__(self, obj):
        super().__init__(obj)

    def __mul__(self, other):
        if isinstance(other, float):
            return FloatVector([i * other for i in self])

    def __truediv__(self, other):
        if isinstance(other, float):
            return r.sapply(self, "*", other)
        elif isinstance(other, int):
            return r.sapply(self, "*", other)

    __rmul__ = __mul__


def is_arr_of_type(arr, check_type):
    if isinstance(arr, list):
        return all(isinstance(x, check_type) for x in arr)
    return False


def is_int_arr(arg):
    return is_arr_of_type(arg, int)


def none_to_null(obj):
    return r('NULL')


def range_to_vec(obj):
    return IntVector(list(obj))


def list_to_vec(obj):
    if is_int_arr(obj):
        return IntVector(list(obj))
    else:
        return robjects.ListVector(list(obj))


def dict_to_vec(obj):
    obj = {str(k): obj[k] for k in obj}
    return r_list(**obj)


def pyssa_to_rssa(obj):
    return obj.obj


pyRssa_converter = conversion.Converter('pyrssa converter')
pyRssa_converter.py2rpy.register(type(None), none_to_null)
pyRssa_converter.py2rpy.register(range, range_to_vec)
pyRssa_converter.py2rpy.register(list, list_to_vec)
pyRssa_converter.py2rpy.register(dict, dict_to_vec)
pyRssa_converter.py2rpy.register(SSA, pyssa_to_rssa)
conversion_rules = default_converter + pyRssa_converter + numpy_converter + pandas_converter
conversion.set_conversion(conversion_rules)


def data(ds_name):
    return read_csv(f'{os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")}/{ds_name}.csv')


def time(ds):
    if isinstance(ds, str):
        return r(f'time({ds})')
    else:
        return r.time(ds)


def ds_cols(ds, *col_names):
    return ds.rx(True, robjects.StrVector(col_names))


def ds_rows(ds, *row_names):
    return ds.rx(robjects.StrVector(row_names), True)


def seed(value):
    r('set.seed')(value)


def window(ds, start=None, end=None):
    if isinstance(ds, str):
        return r(f'window({ds}, start = {start}, end = {end})')
    else:
        return r.window(ds, start=start, end=end)


def r_list(*args, **kwargs):
    if kwargs:
        for k in kwargs:
            if is_int_arr(kwargs[k]):
                kwargs[k] = IntVector(kwargs[k])
        return robjects.vectors.ListVector(kwargs)
    else:
        res = robjects.ListVector.from_length(len(args))
        for i in range(len(args)):
            res[i] = IntVector(args[i])
        return res


def rplot(ds, **kwargs):
    res = r.plot(ds, **kwargs)
    input()
    return res


def seq(f, t):
    return IntVector(list(range(f, t + 1)))


def parestimate(ds, group, method):
    return r.parestimate(ds, group=group, method=method)


def ssa(ds, L, kind):
    return SSA(ds, L, kind)


def reconstruct(ds, groups):
    return Reconstruction(ds, groups)


def wcor(ds, groups=range(1, 50)):
    return WCorMatrix(ds, groups)


def rnorm(*args, **kwargs):
    return r.rnorm(args, kwargs)


def mean(*args):
    return r.mean(*args)


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


def replicate(times, func, *args):
    result = func(*args)
    for i in range(1, times):
        result = r.cbind(result, func(*args))
    return result


def set_names(vec, names):
    vec = robjects.FloatVector(vec)
    vec.names = names
    return vec


# Plot functions


# Deprecated
def vector_plot_2(dt, idx):
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
            print(row, col, eig_n)
            ax[row, col].plot(range(len(dt.U[idx[eig_n]])), dt.U[idx[eig_n]], label=eig_n)

            ax[row, col].set_xticks([])
            ax[row, col].set_yticks([])
            ax[row, col].legend()
            eig_n += 1
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def vector_plot(dt: SSA, idx, contrib=True, layout=None):
    if idx is None:
        idx = range(1, len(dt.U) + 1)
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
    cols = 4
    rows = int(np.ceil(len(idx) / cols))
    fig = plt.figure(figsize=(cols + 1, rows + 1))
    fig.tight_layout(h_pad=1)
    gs = gridspec.GridSpec(rows, cols, width_ratios=[1, 1, 1, 1], figure=fig, wspace=0, hspace=0.5)

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
        ax.set_aspect(aspect='auto', adjustable='box')
    plt.show()


def xyplot(dt: Reconstruction, x, add_residuals, add_original, layout, superpose):

    if x is None:
        x = range(len(dt.series))

    if superpose:
        fig, ax = plt.subplots()
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
                # TODO: Normal warning class
                raise IndexError(f"Layout size {layout} is less than count of plotting series ({cnt}).")
            rows = layout[0]
            cols = layout[1]
        else:
            rows = cnt
            cols = 1
        fig = plt.figure(figsize=(2 * (cols + 1), 2 * (rows + 1)))
        gs = gridspec.GridSpec(rows, cols)
        fig.suptitle("Reconstructed series")
        ax = None

        if add_original:
            ax = fig.add_subplot(gs[0])
            ax.plot(x, dt.series)
            ax.set_title("Original")

        for i in range(add_original, cnt - add_residuals):
            ax = fig.add_subplot(gs[i], sharex=ax)
            name = dt.names[i - add_original]
            ax.plot(x, getattr(dt, name))
            ax.set_title(name)

        if add_residuals:
            ax = fig.add_subplot(gs[cnt - 1], sharex=ax)
            ax.plot(x, dt.residuals)
            ax.set_title("Residuals")

        fig.tight_layout()
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
        ticks = scales
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
    elif type(ts) == Reconstruction or method == "xyplot":
        return xyplot(ts, x, add_residuals, add_original, layout, superpose)
    elif type(ts) == SSA:
        return sigma_plot(ts)
    elif type(ts) == WCorMatrix:
        return wcor_plot(ts, scales)

