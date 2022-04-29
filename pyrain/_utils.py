import pandas
from typing import Union


def _to_timestamp(freq: Union[str, pandas.Timedelta]) -> pandas.Timestamp:
    """

    :param freq: pandas offset string
    """

    origin = pandas.to_datetime("2017-01-01")
    if isinstance(freq, str):
        return origin + pandas.tseries.frequencies.to_offset(freq)
    elif isinstance(freq, pandas.Timedelta):
        return origin + freq
    else:
        raise ValueError("input should either be valid offset alias or Timedelta")


def _get_water_year(idx: pandas.DatetimeIndex, year_start: int) -> pandas.Index:
    """

    :param idx: datetime index
    :param year_start: month when the water year starts
    :return:
    """
    water_year = idx.map(lambda x: x.year if (x.month < year_start) | (year_start == 1) else x.year + 1)
    return water_year
