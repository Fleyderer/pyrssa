import pandas as pd
from rpy2 import robjects
from pyrssa.classes.SSA import SSA
import rpy2.robjects.packages as rpackages
from rpy2.robjects import conversion
from pyrssa.indexing import get_time_index, make_time_index
from functools import cached_property
import numpy as np
from typing import Literal

r_ssa = rpackages.importr('Rssa')
frc = rpackages.importr("forecast")


class BaseForecast:
    """@DynamicAttrs"""

    def __new__(cls, series, f_obj, groups,
                only_new, drop, drop_attributes, reverse=False):
        cls.obj = f_obj
        cls._only_new = only_new
        cls._reverse = reverse
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
                self._forecast_datetime_index = make_time_index(len(f_series), self._datetime_index,
                                                                only_new=self._only_new,
                                                                reverse=self._reverse)
            f_series.index = self._forecast_datetime_index
        return f_series

    @cached_property
    def df(self):
        return pd.DataFrame({name: getattr(self, name) for name in self.names})

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
        elif isinstance(item, int):
            return getattr(self, self.names[item])

    def __str__(self):
        with pd.option_context('display.float_format', '{:.3f}'.format):
            return "\n".join([self.__class__.__name__] +
                             [self.df.__str__()])

    def __repr__(self):
        return self.__str__()


class RForecast(BaseForecast):

    def __new__(cls, x: SSA, groups, length, base, only_new, reverse, drop, drop_attributes, cache, **kwargs):
        rf_obj = r_ssa.rforecast(x, groups, **{"len": length}, base=base, **{"only.new": only_new}, reverse=reverse,
                                 drop=drop, **{"drop.attributes": drop_attributes}, cache=cache, **kwargs)
        return super().__new__(cls, series=x.series, f_obj=rf_obj, groups=groups, only_new=only_new, reverse=reverse,
                               drop=drop, drop_attributes=drop_attributes)


class VForecast(BaseForecast):

    def __new__(cls, x: SSA, groups, length, only_new, drop, drop_attributes, **kwargs):
        vf_obj = r_ssa.vforecast(x, groups, **{"len": length}, **{"only.new": only_new},
                                 drop=drop, **{"drop_attributes": drop_attributes}, **kwargs)

        return super().__new__(cls, series=x.series, f_obj=vf_obj, groups=groups, only_new=only_new,
                               drop=drop, drop_attributes=drop_attributes)


class BForecast(BaseForecast):

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


class Forecast:

    def __init__(self, x: SSA, groups, length=1, method: Literal["recurrent", "vector"] = "recurrent",
                 interval: Literal["none", "confidence", "prediction"] = "none",
                 only_intervals=True, direction: Literal["column", "row"] = "column",
                 drop=True, drop_attributes=False, cache=True, **kwargs):
        self.series = x.series
        self.model = x
        self.obj = r_ssa.forecast_1d_ssa(object=x, groups=groups, len=length, method=method, interval=interval,
                                         **{"only.intervals": only_intervals}, direction=direction, drop=drop,
                                         **{"drop.attributes": drop_attributes}, cache=cache, **kwargs)
        self._index = make_time_index(length, self.series.index, only_new=True)

    @cached_property
    def method(self):
        return self.obj.rx("method")[0]

    @cached_property
    def fitted(self):
        return pd.Series(np.asarray(self.obj.rx("fitted")[0]), index=self.series.index)

    @cached_property
    def residuals(self):
        return pd.Series(np.asarray(self.obj.rx("residuals")[0]), index=self.series.index)

    @cached_property
    def mean(self):
        return pd.Series(np.asarray(self.obj.rx("mean")[0]), index=self._index)

    @cached_property
    def lower(self):
        return pd.Series(np.asarray(self.obj.rx("lower")[0]), index=self._index)

    @cached_property
    def upper(self):
        return pd.Series(np.asarray(self.obj.rx("upper")[0]), index=self._index)

    @cached_property
    def level(self):
        obj = np.asarray(self.obj.rx("level")[0])[0]
        return int(obj) if obj % 1 == 0 else obj

    @cached_property
    def df(self):
        df = pd.DataFrame(zip(self.mean, self.lower, self.upper),
                          columns=["Point Forecast", f"Lo {self.level}", f"Hi {self.level}"],
                          index=self.mean.index)
        return df

    def __str__(self):
        return self.df.__str__()

    def __repr__(self):
        return self.__str__()
