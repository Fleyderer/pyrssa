import pandas as pd
import numpy as np
from rpy2 import robjects
from pyrssa.classes.SSA import SSABase
from pyrssa.classes.AutoSSA import GroupPgram, GroupWCor
from pyrssa.indexing import get_time_index
import rpy2.robjects.packages as rpackages
from typing import Union
from collections.abc import Iterable
from functools import cached_property

r_ssa = rpackages.importr('Rssa')


class Reconstruction:
    """@DynamicAttrs"""

    def __init__(self, x: SSABase, groups: Iterable,
                 drop_attributes=False, cache=True):
        if isinstance(groups, GroupPgram) or isinstance(groups, GroupWCor):
            groups = dict(zip(groups.names, groups.groups))
        self.obj = r_ssa.reconstruct(x=x, groups=groups, **{"drop.attributes": drop_attributes}, cache=cache)
        self._x = x
        self.names = list(robjects.r.names(self.obj))
        self._datetime_index = get_time_index(x.series)

    @cached_property
    def series(self):
        return self._x.series

    @cached_property
    def residuals(self):
        return np.asarray(robjects.r.attr(self.obj, "residuals"))

    def __getattribute__(self, item):
        try:
            return object.__getattribute__(self, item)
        except AttributeError:
            if item in self.names:
                series = pd.Series(self.obj.rx(item)[0])
                series.name = item
                if self._datetime_index is not None:
                    series.index = self._datetime_index
                else:
                    series.index = self.series.index
                setattr(self, item, series)
                return series
            else:
                raise AttributeError

    def __getitem__(self, item):
        if isinstance(item, str) and item in self.keys():
            return getattr(self, item)
        else:
            raise AttributeError(f"This reconstruction object does not have series with name '{item}'.")

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
