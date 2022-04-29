import pandas
import numpy
from scipy.stats import gamma, genextreme, lognorm, kappa3, genpareto, weibull_min, pearson3, expon, beta
from enum import Enum


class _LogPearson3:
    @staticmethod
    def fit(data):
        data = numpy.log10(data)
        return pearson3.fit(data)

    @staticmethod
    def rvs(skew, loc=0, scale=1, size=1):
        data = pearson3.rvs(skew=skew, loc=loc, scale=scale, size=size)
        return numpy.power(10, data)


class Dists(Enum):
    beta = beta
    expon = expon
    gamma = gamma
    genpareto = genpareto
    gev = genextreme
    lognorm = lognorm
    logpearson3 = _LogPearson3
    kappa3 = kappa3
    pearson3 = pearson3
    weibull_min = weibull_min


def _get_dist_func(dist):
    try:
        dist_func = Dists[dist].value
    except KeyError as err:
        print(f"'dist' {dist} is not valid. 'dist' has to be in {[e.name for e in Dists]}")
        raise KeyError(err)

    return dist_func


def fit_dist(data: pandas.Series, dist: str):
    dist_func = _get_dist_func(dist)
    return dist_func.fit(data)


def get_random_value(params: tuple, dist: str, size: int):
    dist_func = _get_dist_func(dist)
    return dist_func.rvs(*params, size=size)
