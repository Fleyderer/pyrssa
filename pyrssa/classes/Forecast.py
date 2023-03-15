import pandas as pd
from rpy2 import robjects
from pyrssa.classes.SSA import SSA
import rpy2.robjects.packages as rpackages
from rpy2.robjects import conversion
from pyrssa.conversion import get_time_index, make_time_index
import numpy as np

r_ssa = rpackages.importr('Rssa')


class RForecast:

    def __init__(self, x: SSA, groups, length, base, only_new, reverse, drop, drop_attributes, cache, **kwargs):
        self.obj = r_ssa.rforecast(x, groups, **{"len": length}, base=base, **{"only.new": only_new}, reverse=reverse,
                                   drop=drop, **{"drop.attributes": drop_attributes}, cache=cache, **kwargs)

        self.names = robjects.r.names(self.obj)

        time_index = get_time_index(x.series)

        for name in self.names:
            series = pd.Series(self.obj.rx(name)[0])
            if time_index is not None:
                series.index = make_time_index(series, time_index, only_new=only_new)
            setattr(self, name, series)

    def __getitem__(self, item):
        if isinstance(item, str):
            return getattr(self, item)

    def __str__(self):
        return str(self.obj)

    def __repr__(self):
        return self.__str__()


class VForecast:

    def __init__(self, x: SSA, groups, length, only_new, drop, drop_attributes, **kwargs):
        self.obj = r_ssa.vforecast(x, groups, **{"len": length}, **{"only.new": only_new},
                                   drop=drop, **{"drop_attributes": drop_attributes}, **kwargs)

        self.names = robjects.r.names(self.obj)

        time_index = get_time_index(x.series)

        for name in self.names:
            series = pd.Series(self.obj.rx(name)[0])
            if time_index is not None:
                series.index = make_time_index(series, time_index, only_new=only_new)
            setattr(self, name, series)

    def __getitem__(self, item):
        if isinstance(item, str):
            return getattr(self, item)

    def __str__(self):
        return str(self.obj)

    def __repr__(self):
        return self.__str__()


class BForecast:

    def __init__(self, x: SSA, groups, length, r, level, kind, interval, only_new, only_intervals,
                 drop, drop_attributes, cache, **kwargs):
        self.obj = r_ssa.bforecast(x, groups, **{"len": length}, R=r, level=level, type=kind,
                                   interval=interval, **{"only.new": only_new}, **{"only.intervals": only_intervals},
                                   drop=drop, **{"drop.attributes": drop_attributes}, cache=cache, **kwargs)

        time_index = get_time_index(x.series)

        self.names = robjects.r.names(self.obj)
        with conversion.localconverter(robjects.default_converter):
            for name in self.names:
                forecast_df = pd.DataFrame(np.asmatrix(self.obj.rx2(name)), columns=list(self.obj.rx2(name).colnames))
                if time_index is not None:
                    forecast_df.index = make_time_index(forecast_df, time_index, only_new=only_new)
                setattr(self, name, forecast_df)

    def __getitem__(self, item):
        if isinstance(item, str):
            return getattr(self, item)

    def __str__(self):
        return str(self.obj)

    def __repr__(self):
        return self.__str__()
