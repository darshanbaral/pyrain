import pandas
from scipy.stats import gamma, genextreme, lognorm
from enum import Enum


class Dists(Enum):
    gamma = gamma
    gev = genextreme
    lognorm = lognorm


def _get_dist_func(dist):
    try:
        dist_func = Dists[dist].value
    except KeyError as err:
        print("'dist' {dist_name} is not valid. 'dist' has to be in ['gamma', 'gev', 'lognorm']".format(dist_name=dist))
        raise KeyError(err)

    return dist_func


def fit_dist(data: pandas.Series, dist: str):
    dist_func = _get_dist_func(dist)
    return dist_func.fit(data)


def get_random_value(params: tuple, dist: str, size: int):
    dist_func = _get_dist_func(dist)
    return dist_func.rvs(*params, size=size)
