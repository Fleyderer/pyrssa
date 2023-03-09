from rpy2 import robjects
from pyrssa.classes.SSA import SSA
import rpy2.robjects.packages as rpackages
import numpy as np

r_ssa = rpackages.importr('Rssa')


class RForecast:

    def __init__(self, ds: SSA, groups, length, base, only_new, drop, drop_attributes, cache, ** kwargs):
        self.obj = r_ssa.rforecast(ds, groups, len=length, base=base, only_new=only_new,
                                   drop=drop, drop_attributes=drop_attributes, cache=cache, **kwargs)
        self.names = robjects.r.names(self.obj)
        for name in self.names:
            # TODO: add auto conversion from R ts to pandas Series with dates.
            setattr(self, name, self.obj.rx(name)[0])


class VForecast:

    def __init__(self, ds, groups, length, only_new, drop, drop_attributes, ** kwargs):
        self.obj = r_ssa.vforecast(ds, groups, len=length, only_new=only_new,
                                   drop=drop, drop_attributes=drop_attributes, **kwargs)
        self.names = robjects.r.names(self.obj)
        for name in self.names:
            # TODO: add auto conversion from R ts to pandas Series with dates.
            setattr(self, name, self.obj.rx(name)[0])


# TODO: Make working version of bforecast
class BForecast:

    def __init__(self, ds, groups, length, r, level, kind, interval, only_new, only_intervals,
                 drop, drop_attributes, cache, ** kwargs):
        self.obj = r_ssa.bforecast(ds, groups, len=length, R=r, level=level, type=kind,
                                   interval=interval, only_new=only_new, only_intervals=only_intervals,
                                   drop=drop, drop_attributes=drop_attributes, cache=cache, **kwargs)
        self.names = robjects.r.names(self.obj)
        for name in self.names:
            # TODO: add auto conversion from R ts to pandas Series with dates.
            setattr(self, name, self.obj.rx(name)[0])
