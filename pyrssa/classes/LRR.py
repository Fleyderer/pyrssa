import numpy as np
from rpy2 import robjects
from pyrssa.classes.SSA import SSABase
import rpy2.robjects.packages as rpackages
from typing import Literal

r_ssa = rpackages.importr("Rssa")
base = rpackages.importr("base")
assign_attr = base._env['attr<-']


def _get_lrr(x, groups=None, reverse=False, drop=True,
             eps=np.sqrt(np.finfo(float).eps), raw=False, orthonormalize=True):
    if isinstance(x, SSABase):
        return r_ssa.lrr(x, groups=groups, reverse=reverse, drop=drop)
    else:
        raise NotImplementedError("Not defined by now")


class BaseLRR(np.ndarray):

    def __new__(cls, lrr_obj=None, x: SSABase = None, groups=None, reverse=False, drop=True,
                eps=np.sqrt(np.finfo(float).eps), raw=False, orthonormalize=True):

        if groups is None:
            groups = range(1, min(x.nsigma(), x.nu()) + 1)

        if lrr_obj is not None:
            obj = lrr_obj
        else:
            obj = _get_lrr(x=x, groups=groups, reverse=reverse, drop=drop,
                           eps=eps, raw=raw, orthonormalize=orthonormalize)
        result = np.asarray(obj).view(cls)
        result._obj = obj
        return result

    def __array_wrap__(self, array, context=None, *args, **kwargs):
        self._obj = robjects.FloatVector(array)
        self._obj = assign_attr(self._obj, "class", value="lrr")
        return super().__array_wrap__(np.asarray(self._obj), array, context)

    def __array_finalize__(self, obj, *args, **kwargs):
        if obj is None:
            return
        self._obj = getattr(obj, '_obj', None)

    def roots(self, method: Literal["companion", "polyroot"] = "companion"):
        return np.asarray(r_ssa.roots(self._obj, method=method))

    def __str__(self):
        return "lrr: " + str(np.asarray(self._obj))

    def __repr__(self):
        return self.__str__()


class LRR:
    """@DynamicAttrs"""

    def __init__(self, x: SSABase, groups=None, reverse=False, drop=True,
                 eps=np.sqrt(np.finfo(float).eps), raw=False, orthonormalize=True):

        if groups is None:
            groups = range(1, min(x.nsigma(), x.nu()) + 1)

        self.obj = _get_lrr(x=x, groups=groups, reverse=reverse, drop=drop, eps=eps,
                            raw=raw, orthonormalize=orthonormalize)

        self.names = list(robjects.r.names(self.obj))

    def __getattribute__(self, item) -> BaseLRR:
        try:
            return object.__getattribute__(self, item)
        except AttributeError:
            if item in self.names:
                res = BaseLRR(lrr_obj=self.obj.rx(item)[0])
                setattr(self, item, res)
                return res
            else:
                raise AttributeError(f"This LRR object does not have group with name '{item}'.")

    def __getitem__(self, item) -> BaseLRR:
        if isinstance(item, str) and item in self.keys():
            return getattr(self, item)
        else:
            raise AttributeError(f"This LRR object does not have group with name '{item}'.")

    def keys(self):
        return self.names

    def values(self):
        return [self.__getitem__(name) for name in self.names]

    def items(self):
        return zip(self.keys(), self.values())

    def __str__(self):
        return str(self.obj)

    def __repr__(self):
        return self.__str__()
