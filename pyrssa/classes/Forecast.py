import pandas as pd
from rpy2 import robjects
from pyrssa.classes.SSA import SSA
from pyrssa.classes.Resonstruction import Reconstruction
import rpy2.robjects.packages as rpackages
import numpy as np

r_ssa = rpackages.importr('Rssa')


class RForecast:

    def __init__(self, ds: SSA, groups, length, base, only_new, reverse, drop, drop_attributes, cache, **kwargs):
        self.obj = r_ssa.rforecast(ds, groups, len=length, base=base, only_new=only_new, reverse=reverse,
                                   drop=drop, drop_attributes=drop_attributes, cache=cache, **kwargs)
        self.names = robjects.r.names(self.obj)

        # TODO: Find a reason, why "only.new" parameter is ignored via rpy2
        if not only_new:
            if base == "reconstructed":
                base_series = Reconstruction(ds, groups=groups)
            else:
                base_series = ds.F.iloc[:, 0]
        else:
            base_series = []

        for name in self.names:
            # TODO: add auto conversion from R ts to pandas Series with dates.
            add_series = base_series[name] if base == "reconstructed" else base_series
            setattr(self, name, np.concatenate((add_series, self.obj.rx(name)[0])))

    def __getitem__(self, item):
        if isinstance(item, str):
            return getattr(self, item)

    def __str__(self):
        return str(self.obj)

    def __repr__(self):
        return self.__str__()


class VForecast:

    def __init__(self, ds, groups, length, only_new, drop, drop_attributes, ** kwargs):
        self.obj = r_ssa.vforecast(ds, groups, len=length, only_new=only_new,
                                   drop=drop, drop_attributes=drop_attributes, **kwargs)

        base_series = Reconstruction(ds, groups=groups) if not only_new else []
        self.names = robjects.r.names(self.obj)
        for name in self.names:
            # TODO: add auto conversion from R ts to pandas Series with dates.
            if only_new:
                setattr(self, name, self.obj.rx(name)[0])
            else:
                setattr(self, name, np.concatenate((base_series[name], self.obj.rx(name)[0])))

    def __getitem__(self, item):
        if isinstance(item, str):
            return getattr(self, item)

    def __str__(self):
        return str(self.obj)

    def __repr__(self):
        return self.__str__()


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

    def __getitem__(self, item):
        if isinstance(item, str):
            return getattr(self, item)

    def __str__(self):
        return str(self.obj)

    def __repr__(self):
        return self.__str__()
