import pandas as pd
import numpy as np
from typing import Literal
from rpy2 import robjects
import rpy2.robjects.packages as rpackages
from pyrssa.classes.SSA import SSA
from pyrssa.classes.WCorMatrix import WCorMatrix
from rpy2.robjects import conversion

r_ssa = rpackages.importr('Rssa')
ssa_get = robjects.r('utils::getFromNamespace("$.ssa", "Rssa")')


class AutoGroup:

    def __init__(self, obj):
        self.obj = obj
        self.names = list(robjects.r.names(self.obj))
        self.groups = []
        for i, name in enumerate(self.names):
            group = np.asarray(self.obj.rx(name)[0])
            self.groups.append(group)
            setattr(self, name, group)

    def __getitem__(self, item):
        if isinstance(item, str):
            return getattr(self, item)
        elif isinstance(item, int):
            return self.groups[item]

    def keys(self):
        return self.names

    def values(self):
        return [self.__getitem__(name) for name in self.names]

    def items(self):
        return zip(self.keys(), self.values())

    def __len__(self):
        return len(self.names)

    def __str__(self):
        return str(self.obj)

    def __repr__(self):
        return self.__str__()


class GroupPgram(AutoGroup):

    def __init__(self, x: SSA, groups=None, base: Literal["series", "eigen", "factor"] = "series", freq_bins=2,
                 threshold=0, method: Literal["constant", "linear"] = "constant", drop=True, **kwargs):

        if groups is None:
            groups = range(1, min(x.nsigma(), x.nu()) + 1)

        super().__init__(obj=r_ssa.grouping_auto_pgram(x=x, groups=groups, base=base, **{"freq.bins": freq_bins},
                                                       threshold=threshold, method=method, drop=drop, **kwargs))
        with conversion.localconverter(robjects.default_converter):
            contributions_obj = robjects.r.attr(self.obj, "contributions")
            self.contributions = pd.DataFrame(np.asmatrix(contributions_obj), columns=list(contributions_obj.colnames))

        self.type = robjects.r.attr(self.obj, "type")
        self.threshold = robjects.r.attr(self.obj, "threshold")


class GroupWCor(AutoGroup):

    def __init__(self, x: SSA, groups=None, nclust=None, **kwargs):
        if groups is None:
            groups = range(1, min(x.nsigma(), x.nu()) + 1)
        if nclust is None:
            nclust = len(groups) // 2

        super().__init__(obj=r_ssa.grouping_auto_wcor(x=x, groups=groups, nclust=nclust, **kwargs))

        self.hclust = robjects.r.attr(self.obj, "hclust")
        self.wcor = WCorMatrix(np.asmatrix(robjects.r.attr(self.obj, "wcor")))
