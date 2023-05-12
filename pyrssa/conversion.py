import pandas
from rpy2 import robjects
import rpy2.robjects.conversion as conversion
from rpy2.robjects import default_converter
from rpy2.robjects.numpy2ri import converter as numpy_converter
from rpy2.robjects.pandas2ri import converter as pandas_converter
import numpy as np
import pandas as pd
from pyrssa.classes.SSA import SSABase
from typing import Union


def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False


def is_list(obj):
    return is_iterable(obj) and not isinstance(obj, (str, dict))


def is_arr_of_type(arr, check_type, allow_inf=True, allow_na=True):
    if isinstance(arr, list) or isinstance(arr, tuple) or isinstance(arr, np.ndarray):
        for x in arr:
            if not isinstance(x, check_type):
                if not hasattr(x, '__iter__'):
                    if allow_na and not np.isnan(x) or allow_inf and not np.isinf(x):
                        return False
                else:
                    return False
        return True
    else:
        return False


def is_arr_of_types(obj, check_types, allow_inf=True, allow_na=True):
    if is_list(obj):
        return all([False if is_list(x) else
                    isinstance(x, check_types)
                    or (allow_na and np.isnan(x))
                    or (allow_inf and np.isinf(x)) for x in obj])


def is_int_arr(obj):
    return isinstance(obj, range) or is_arr_of_types(obj, (int, np.int32, np.int64))


def is_float_arr(obj):
    return is_arr_of_types(obj, (float, np.float32, np.float64))


def none_to_null(_):
    return robjects.r('NULL')


def range_to_vec(obj):
    return robjects.IntVector(list(obj))


def collection_conversion(obj):
    if is_list(obj):
        if is_int_arr(obj):
            return robjects.IntVector(list(obj))
        elif is_float_arr(obj):
            return robjects.FloatVector(list(obj))
        else:
            result = robjects.ListVector.from_length(len(obj))
            for i, x in enumerate(obj):
                result[i] = collection_conversion(x)
            return result
    elif isinstance(obj, dict):
        return robjects.vectors.ListVector({str(k): collection_conversion(obj[k]) for k in obj})
    else:
        return obj


def convert(obj):
    if is_int_arr(obj):
        return robjects.IntVector(obj)


def pyrssa_to_rssa(obj):
    return obj.obj


def get_time_index(series):
    if isinstance(series, pd.DataFrame) or isinstance(series, pd.Series):
        if isinstance(series.index, pd.DatetimeIndex):
            return series.index
    return None


def make_time_index(length: int, time_index: pd.DatetimeIndex,
                    only_new=False, reverse=False):
    # if only_new is False, we have to ignore old series, when creating new indices
    periods = length - (not only_new) * len(time_index) + 1
    if reverse:
        new = pd.date_range(end=min(time_index), freq=time_index.freqstr, periods=periods, inclusive="left")
    else:
        new = pd.date_range(max(time_index), freq=time_index.freqstr, periods=periods,
                            inclusive="right")
    if only_new:
        return new
    else:
        return time_index.union(new)


pyrssa_converter = conversion.Converter('pyrssa converter')
pyrssa_converter.py2rpy.register(type(None), none_to_null)
pyrssa_converter.py2rpy.register(range, collection_conversion)
pyrssa_converter.py2rpy.register(tuple, collection_conversion)
pyrssa_converter.py2rpy.register(list, collection_conversion)
pyrssa_converter.py2rpy.register(dict, collection_conversion)
pyrssa_converter.py2rpy.register(SSABase, pyrssa_to_rssa)

# pyrssa_conversion_rules = default_converter + pyrssa_converter + numpy_converter + pandas_converter
pyrssa_conversion_rules = pandas_converter + numpy_converter + default_converter + pyrssa_converter
