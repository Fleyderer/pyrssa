import pandas as pd


def get_time_index(series):
    if isinstance(series, pd.DataFrame) or isinstance(series, pd.Series):
        if isinstance(series.index, pd.DatetimeIndex):
            return series.index
    return None


def make_time_index(length: int, time_index: pd.DatetimeIndex,
                    only_new=False, reverse=False):
    # if only_new is False, we have to ignore old series, when creating new indices
    periods = length - (not only_new) * len(time_index) + 1
    if reverse:
        new = pd.date_range(end=min(time_index), freq=time_index.freqstr, periods=periods, inclusive="left")
    else:
        new = pd.date_range(max(time_index), freq=time_index.freqstr, periods=periods,
                            inclusive="right")
    if only_new:
        return new
    else:
        return time_index.union(new)