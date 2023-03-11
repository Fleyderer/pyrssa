import pandas
from rpy2 import robjects
from pyrssa.classes.SSA import SSA
import rpy2.robjects.packages as rpackages

r_ssa = rpackages.importr('Rssa')


class Reconstruction:

    def __init__(self, x: SSA, groups, drop_attributes=False, cache=True):
        self.obj = r_ssa.reconstruct(x=x, groups=groups, drop_attributes=drop_attributes, cache=cache)
        self.series = x.F
        self.residuals = robjects.r.attr(self.obj, "residuals")
        self.names = robjects.r.names(self.obj)
        for name in self.names:
            setattr(self, name, self.obj.rx(name)[0])

    def __getitem__(self, item):
        if isinstance(item, str):
            return getattr(self, item)

    def __str__(self):
        return str(self.obj)

    def __repr__(self):
        return self.__str__()


