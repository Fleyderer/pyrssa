from rpy2 import robjects
import rpy2.robjects.conversion as conversion
from rpy2.robjects import default_converter
from rpy2.robjects.numpy2ri import converter as numpy_converter
from rpy2.robjects.pandas2ri import converter as pandas_converter
from pyrssa import SSA


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

    __rmul__ = __mul__


def is_arr_of_type(arr, check_type):
    if isinstance(arr, list):
        return all(isinstance(x, check_type) for x in arr)
    return False


def is_int_arr(arg):
    return is_arr_of_type(arg, int)


def none_to_null(obj):
    return robjects.r('NULL')


def range_to_vec(obj):
    return IntVector(list(obj))


def list_to_vec(obj):
    if is_int_arr(obj):
        return IntVector(list(obj))
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


pyrssa_converter = conversion.Converter('pyrssa converter')
pyrssa_converter.py2rpy.register(type(None), none_to_null)
pyrssa_converter.py2rpy.register(range, range_to_vec)
pyrssa_converter.py2rpy.register(list, list_to_vec)
pyrssa_converter.py2rpy.register(dict, dict_to_vec)
pyrssa_converter.py2rpy.register(SSA, pyrssa_to_rssa)

pyrssa_conversion_rules = default_converter + pyrssa_converter + numpy_converter + pandas_converter
