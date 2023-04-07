import pandas as pd
from rpy2 import robjects
from pyrssa.classes.SSA import SSA
import rpy2.robjects.packages as rpackages
from rpy2.robjects import conversion
from pyrssa.conversion import get_time_index, make_time_index
import numpy as np

r_ssa = rpackages.importr('Rssa')


class Forecast:
    """@DynamicAttrs"""

    def __new__(cls, series, f_obj, groups,
                only_new, drop, drop_attributes):
        cls.obj = f_obj
        cls._only_new = only_new
        cls._drop = drop
        cls._drop_attributes = drop_attributes
        cls._datetime_index = get_time_index(series)
        cls._forecast_datetime_index = None

        instance = super().__new__(cls)

        if not drop or drop and len(groups) != 1:
            cls.names = robjects.r.names(cls.obj)
        else:
            cls.names = None
            instance = cls._finalize_forecast(
                instance, cls._get_forecast(
                    instance, f"{series.name} {cls.__name__}"))

        return instance

    def _get_forecast(self, name):
        if self.names is None:
            result = pd.Series(self.obj)
        else:
            result = pd.Series(self.obj.rx(name)[0])
        result.name = name
        return result

    def _finalize_forecast(self, f_series):
        if self._datetime_index is not None and not self._drop_attributes:
            if self._forecast_datetime_index is None:
                self._forecast_datetime_index = make_time_index(f_series, self._datetime_index,
                                                                only_new=self._only_new)
            f_series.index = self._forecast_datetime_index
        return f_series

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

    def __str__(self):
        with pd.option_context('display.float_format', '{:.3f}'.format):
            return "\n".join([self.__class__.__name__] +
                             [pd.DataFrame({name: getattr(self, name)
                                            for name in self.names}).__str__()])

    def __repr__(self):
        return self.__str__()


class RForecast(Forecast):

    def __new__(cls, x: SSA, groups, length, base, only_new, reverse, drop, drop_attributes, cache, **kwargs):
        rf_obj = r_ssa.rforecast(x, groups, **{"len": length}, base=base, **{"only.new": only_new}, reverse=reverse,
                                 drop=drop, **{"drop.attributes": drop_attributes}, cache=cache, **kwargs)
        return super().__new__(cls, series=x.series, f_obj=rf_obj, groups=groups, only_new=only_new,
                               drop=drop, drop_attributes=drop_attributes)


class VForecast(Forecast):

    def __new__(cls, x: SSA, groups, length, only_new, drop, drop_attributes, **kwargs):
        vf_obj = r_ssa.vforecast(x, groups, **{"len": length}, **{"only.new": only_new},
                                 drop=drop, **{"drop_attributes": drop_attributes}, **kwargs)

        return super().__new__(cls, series=x.series, f_obj=vf_obj, groups=groups, only_new=only_new,
                               drop=drop, drop_attributes=drop_attributes)


class BForecast(Forecast):

    def __new__(cls, x: SSA, groups, length, r, level, kind, interval, only_new, only_intervals,
                drop, drop_attributes, cache, **kwargs):
        bf_obj = r_ssa.bforecast(x, groups, **{"len": length}, R=r, level=level, type=kind,
                                 interval=interval, **{"only.new": only_new}, **{"only.intervals": only_intervals},
                                 drop=drop, **{"drop.attributes": drop_attributes}, cache=cache, **kwargs)

        return super().__new__(cls, series=x.series, f_obj=bf_obj, groups=groups, only_new=only_new,
                               drop=drop, drop_attributes=drop_attributes)

    def _get_forecast(self, name):
        with conversion.localconverter(robjects.default_converter):
            if self.names is None:
                result = pd.DataFrame(np.asmatrix(self.obj),
                                      columns=list(self.obj.colnames))
            else:
                result = pd.DataFrame(np.asmatrix(self.obj.rx2(name)),
                                      columns=list(self.obj.rx2(name).colnames))
            return result

    def __str__(self):
        with pd.option_context('display.float_format', '{:.3f}'.format):
            return "\n".join([self.__class__.__name__] +
                             [f"{name}:\n{getattr(self, name)}\n" for name in self.names])
