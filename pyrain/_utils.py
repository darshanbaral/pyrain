import pandas
from typing import Union
from datetime import datetime


def _to_timestamp(freq: Union[str, pandas.Timedelta],
                  origin: datetime = datetime(2017, 1, 1)):
    """

    :param freq:
    :param origin:
    :return:
    """
    if isinstance(freq, str):
        return origin + pandas.tseries.frequencies.to_offset(freq)
    elif isinstance(freq, pandas.Timedelta):
        return origin + freq
    else:
        raise ValueError("input should either be valid offset alias or Timedelta")