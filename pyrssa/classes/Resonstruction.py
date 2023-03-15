import pandas as pd
import numpy as np
from rpy2 import robjects
from pyrssa.classes.SSA import SSABase
from pyrssa.classes.AutoSSA import GroupPgram, GroupWCor
from pyrssa.conversion import get_time_index
import rpy2.robjects.packages as rpackages
from typing import Union

r_ssa = rpackages.importr('Rssa')


class Reconstruction:

    def __init__(self, x: SSABase, groups: Union[list, dict, np.ndarray, GroupPgram, GroupWCor],
                 drop_attributes=False, cache=True):
        if isinstance(groups, GroupPgram) or isinstance(groups, GroupWCor):
            groups = groups.groups
        self.obj = r_ssa.reconstruct(x=x, groups=groups, **{"drop.attributes": drop_attributes}, cache=cache)
        self.series = x.series
        self.residuals = robjects.r.attr(self.obj, "residuals")
        self.names = list(robjects.r.names(self.obj))

        time_index = get_time_index(x.series)

        for name in self.names:
            series = pd.Series(self.obj.rx(name)[0])
            if time_index is not None:
                series.index = time_index
            setattr(self, name, series)

    def __getitem__(self, item):
        if isinstance(item, str):
            return getattr(self, item)

    def __str__(self):
        return str(self.obj)

    def __repr__(self):
        return self.__str__()
