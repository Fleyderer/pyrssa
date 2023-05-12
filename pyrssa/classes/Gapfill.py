import pandas as pd

import os
import platform

if os.environ.get("R_HOME") is None:

    if platform.system() == "Windows":
        path = r"C:\Program Files\R"
        if os.path.exists(path):
            os.environ["R_HOME"] = os.path.join(path, os.listdir(path)[-1])
        else:
            raise FileNotFoundError("R_HOME variable does not exist")
    else:
        raise FileNotFoundError("R_HOME variable does not exist")


from rpy2 import robjects
from pyrssa.classes.SSA import SSA
import rpy2.robjects.packages as rpackages
from rpy2.robjects import conversion
from pyrssa.conversion import get_time_index, make_time_index
from functools import cached_property
import numpy as np
from typing import Literal

r_ssa = rpackages.importr('Rssa')
frc = rpackages.importr("forecast")


class BaseGapfill:
    """@DynamicAttrs"""

    def __new__(cls, series, g_obj, groups, drop, drop_attributes):
        cls.obj = g_obj
        cls._drop = drop
        cls._drop_attributes = drop_attributes
        cls._index = get_time_index(series)

        instance = super().__new__(cls)

        if not drop or drop and len(groups) != 1:
            cls.names = robjects.r.names(cls.obj)
        else:
            cls.names = None
            instance = cls._finalize_gapfill(
                instance, cls._get_gapfill(
                    instance, f"{series.name} {cls.__name__}"))

        return instance

    def _get_gapfill(self, name):
        if self.names is None:
            result = pd.Series(self.obj)
        else:
            result = pd.Series(self.obj.rx(name)[0])
        result.name = name
        return result

    def _finalize_gapfill(self, f_series):
        if self._index is not None and not self._drop_attributes:
            f_series.index = self._index
        return f_series

    @cached_property
    def df(self):
        return pd.DataFrame({name: getattr(self, name) for name in self.names})

    def __getattribute__(self, item):
        try:
            return object.__getattribute__(self, item)
        except AttributeError:
            if item in self.names:
                series = self._finalize_forecast(self._get_forecast(item))
                setattr(self, item, series)
                return series
            else:
                raise AttributeError

    def __getitem__(self, item):
        if isinstance(item, str):
            return getattr(self, item)
        elif isinstance(item, int):
            return getattr(self, self.names[item])

    def __str__(self):
        with pd.option_context('display.float_format', '{:.3f}'.format):
            return "\n".join([self.__class__.__name__] +
                             [self.df.__str__()])

    def __repr__(self):
        return self.__str__()


def _alpha_conversion(func):
    def wrapper(x):
        return robjects.FloatVector(func(int(x[0])))
    return wrapper


def _default_alpha(length):
    return np.linspace(0, 1, num=length)


class Gapfill(BaseGapfill):

    def __new__(cls, x: SSA, groups, base, method, alpha, drop, drop_attributes, cache, **kwargs):

        if not isinstance(alpha, (int, float)):
            if alpha is None:
                alpha = _default_alpha
            alpha = robjects.rinterface.rternalize(_alpha_conversion(alpha))
        g_obj = r_ssa.gapfill(x=x, groups=groups, base=base, method=method, alpha=alpha, **kwargs,
                              drop=drop, **{"drop.attributes": drop_attributes}, cache=cache)
        return super().__new__(cls, series=x.series, g_obj=g_obj, groups=groups,
                               drop=drop, drop_attributes=drop_attributes)




