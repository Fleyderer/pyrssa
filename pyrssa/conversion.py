from rpy2 import robjects
import rpy2.robjects.conversion as conversion
from rpy2.robjects import default_converter
from rpy2.rinterface_lib import sexp
from rpy2.robjects.numpy2ri import converter as numpy_converter
from rpy2.robjects.pandas2ri import converter as pandas_converter
import numpy as np
import pandas as pd
from pyrssa.classes.SSA import SSABase
from typing import Union


class FloatVector(robjects.FloatVector):
    def __init__(self, obj):
        super().__init__(obj)

    def __mul__(self, other):
        if isinstance(other, float):
            return robjects.r.sapply(self, "*", other)
        elif isinstance(other, int):
            return robjects.r.sapply(self, "*", other)

    def __truediv__(self, other):
        if isinstance(other, float):
            return robjects.r.sapply(self, "/", other)
        elif isinstance(other, int):
            return robjects.r.sapply(self, "/", other)

    def __add__(self, other):
        if isinstance(other, float):
            return robjects.r.sapply(self, "+", other)
        elif isinstance(other, int):
            return robjects.r.sapply(self, "+", other)

    def __pow__(self, power, modulo=None):
        if isinstance(power, float) or isinstance(power, int):
            return robjects.r.sapply(self, "^", power)

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
            return robjects.r.sapply(self, "*", other)
        elif isinstance(other, int):
            return robjects.r.sapply(self, "*", other)

    def __pow__(self, power, modulo=None):
        if isinstance(power, float) or isinstance(power, int):
            return robjects.r.sapply(self, "^", power)

    __rmul__ = __mul__


def is_arr_of_type(arr, check_type, allow_inf=True, allow_na=True):
    if isinstance(arr, list) or isinstance(arr, tuple) or isinstance(arr, np.ndarray):
        for x in arr:
            if not isinstance(x, check_type):
                if not hasattr(x, '__iter__'):
                    if allow_na and np.isnan(x) or allow_inf and np.isinf(x):
                        return False
                else:
                    return False
        return True
    else:
        return False


def is_int_arr(arr):
    return is_arr_of_type(arr, int)


def is_float_arr(arr):
    return is_arr_of_type(arr, float)


def is_of_int_lists_arr(arr):
    if isinstance(arr, list):
        for x in arr:
            if not isinstance(x, int) and not is_int_arr(x) and not isinstance(x, range):
                return False
    return True


def none_to_null(obj):
    return robjects.r('NULL')


def range_to_vec(obj):
    return IntVector(list(obj))


def list_to_vec(obj):
    if is_int_arr(obj):
        return IntVector(list(obj))
    elif is_float_arr(obj):
        return FloatVector(list(obj))
    elif is_of_int_lists_arr(obj):
        result = robjects.ListVector.from_length(len(obj))
        for i, x in enumerate(obj):
            if isinstance(x, int):
                result[i] = robjects.IntVector([x])
            else:
                result[i] = robjects.IntVector(list(x))
        return result
    else:
        return robjects.ListVector(list(obj))


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


def dict_to_vec(obj):
    obj = {str(k): obj[k] for k in obj}
    return r_list(**obj)


def pyrssa_to_rssa(obj):
    return obj.obj


def get_time_index(series):
    if isinstance(series, pd.DataFrame) or isinstance(series, pd.Series):
        if isinstance(series.index, pd.DatetimeIndex):
            return series.index
    return None


def make_time_index(series: Union[pd.Series, pd.DataFrame], time_index: pd.DatetimeIndex, only_new=False):
    # if only_new is False, we have to ignore old series, when creating new indices
    periods = len(series) - (not only_new) * len(time_index) + 1
    new = pd.date_range(max(time_index), freq=time_index.freqstr, periods=periods,
                        inclusive="right")
    if only_new:
        return new
    else:
        return time_index.union(new)


pyrssa_converter = conversion.Converter('pyrssa converter')
pyrssa_converter.py2rpy.register(type(None), none_to_null)
pyrssa_converter.py2rpy.register(range, range_to_vec)
pyrssa_converter.py2rpy.register(tuple, list_to_vec)
pyrssa_converter.py2rpy.register(list, list_to_vec)
pyrssa_converter.py2rpy.register(dict, dict_to_vec)
pyrssa_converter.py2rpy.register(SSABase, pyrssa_to_rssa)

# pyrssa_conversion_rules = default_converter + pyrssa_converter + numpy_converter + pandas_converter
pyrssa_conversion_rules = pandas_converter + numpy_converter + default_converter + pyrssa_converter
