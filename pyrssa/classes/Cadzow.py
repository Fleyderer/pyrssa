from rpy2 import robjects
from pyrssa.classes.SSA import SSA
import rpy2.robjects.packages as rpackages
from pyrssa.indexing import get_time_index
import numpy as np
import pandas as pd
from typing import Callable

r_ssa = rpackages.importr('Rssa')


def _norm_conversion(func):
    def wrapper(x):
        return float(func(np.array(x)))

    return wrapper


def _default_norm(x):
    return np.max(np.abs(x))


class Cadzow:
    """@DynamicAttrs"""

    def __new__(cls, x: SSA, rank: int, correct: bool = True, tol: float = 1e-6, maxiter: int = 0,
                norm: Callable = None, trace: bool = False, cache: bool = True, **kwargs):

        if norm is None:
            norm = _default_norm
        norm = robjects.rinterface.rternalize(_norm_conversion(norm))
        cls.obj = r_ssa.cadzow(x=x, rank=rank, correct=correct, tol=tol, maxiter=maxiter,
                               norm=norm, trace=trace, **kwargs, cache=cache)

        return pd.Series(np.asarray(cls.obj), index=get_time_index(x.series))
