from rpy2 import robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import STAP
import rpy2.robjects.conversion as conversion
from rpy2.robjects import default_converter
from rpy2.robjects.numpy2ri import converter as numpy_converter
import matplotlib.pyplot as plt

r = robjects.r
r_ssa = rpackages.importr('Rssa')
r_lattice = rpackages.importr('lattice')
py_list = list
pi = r.pi[0]


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


rssa_pkg = STAP(rssa_plt_1d, "rssa_pkg")


def is_arr_of_type(arr, check_type):
    if isinstance(arr, py_list):
        return all(isinstance(x, check_type) for x in arr)
    return False


def is_int_arr(arg):
    return is_arr_of_type(arg, int)


def none_to_null(obj):
    return r('NULL')


def range_to_vec(obj):
    return IntVector(py_list(obj))


def list_to_vec(obj):
    if is_int_arr(obj):
        return IntVector(py_list(obj))
    else:
        return robjects.ListVector(py_list(obj))


def dict_to_vec(obj):
    obj = {str(k): obj[k] for k in obj}
    return list(**obj)


pyRssa_converter = conversion.Converter('pyRssa converter')
pyRssa_converter.py2rpy.register(type(None), none_to_null)
pyRssa_converter.py2rpy.register(range, range_to_vec)
pyRssa_converter.py2rpy.register(py_list, list_to_vec)
pyRssa_converter.py2rpy.register(dict, dict_to_vec)
conversion_rules = default_converter + pyRssa_converter + numpy_converter
conversion.set_conversion(conversion_rules)


def data(ds_name, package):
    r(f'data("{ds_name}", package = "{package}")')
    return r(ds_name)


def time(ds):
    if isinstance(ds, str):
        return r(f'time({ds})')
    else:
        return r.time(ds)


def cols(ds, *col_names):
    return ds.rx(True, robjects.StrVector(col_names))


def rows(ds, *row_names):
    return ds.rx(robjects.StrVector(row_names), True)


def seed(value):
    r('set.seed')(value)


def window(ds, start=None, end=None):
    if isinstance(ds, str):
        return r(f'window({ds}, start = {start}, end = {end})')
    else:
        return r.window(ds, start=start, end=end)


def ssa(ds, L, kind):
    return r_ssa.ssa(ds, L=L, kind=kind)


def list(*args, **kwargs):
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



def reconstruct(ds, groups):
    return r_ssa.reconstruct(ds, groups=groups)


def plot(ds, **kwargs):
    res = r.plot(ds, **kwargs)
    input()
    return res


def seq(f, t):
    return IntVector(py_list(range(f, t + 1)))


def parestimate(ds, group, method):
    return r.parestimate(ds, group=group, method=method)


def wcor(ds, groups):
    return r.wcor(ds, groups=groups)


def plot_1d(ds, **kwargs):
    rssa_pkg.plot_1d(ds, **kwargs)


def rnorm(*args, **kwargs):
    return r.rnorm(args, kwargs)


def mean(*args):
    return r.mean(*args)


def vforecast(*args, **kwargs):
    return r.vforecast(*args, **kwargs)


def replicate(times, func, *args):
    result = func(*args)
    for i in range(1, times):
        result = r.cbind(result, func(*args))
    return result


def set_names(vec, names):
    vec = robjects.FloatVector(vec)
    vec.names = names
    return vec


def mplot(dt, X=None, add_residuals=False, add_original=False):
    fig, ax = plt.subplots()
    series = r.attr(dt, "series")
    if X is None:
        X = range(len(series))
    if add_original:
        ax.plot(X, series, label="Original")
    ax.plot(X, dt.rx("Trend")[0], label='Trend')
    ax.plot(X, dt.rx("Seasonality")[0], label='Seasonality')
    if add_residuals:
        ax.plot(X, r.attr(dt, "residuals"), label='Residuals')
    ax.legend()
    plt.title(label="Reconstructed series")
    plt.show()

