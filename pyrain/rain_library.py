from multiprocessing import Pool
import numpy
import pandas
import toml
from pathlib import Path
from typing import Union
import math
from . import _stats
from .rain import Rain
from .synthetic_rain import SyntheticRain
from ._utils import _to_timestamp


def load_synthetic_rain(data_path: Union[str, Path],
                        info_path: Union[str, Path] = None):
    """
    load saved synthetic rain data.
    see [`SyntheticRain.save()`](./synthetic_rain.html#pyrain.synthetic_rain.SyntheticRain.save)

    Args:
        data_path: path to the rainfall data csv
        info_path: path to the rainfall data info

    Returns:
        [`SyntheticRain`](./synthetic_rain.html)
    """
    data = pandas.read_csv(data_path, index_col=0)
    data.index = pandas.to_datetime(data.index)

    if info_path is None:
        time_step, block_size, dist_name, dist_params, low, high = [None] * 6
    else:
        info = toml.load(info_path)
        time_step = pandas.Timedelta(info["time_step"])
        block_size = pandas.Timedelta(info["block_size"])
        dist_name = info["dist_name"]
        low = pandas.Timedelta(info["low"])
        high = pandas.Timedelta(info["high"])
        dist_params = tuple(info["dist_params"])

    return SyntheticRain(data, time_step, block_size, dist_name, dist_params, low, high)


class RainLibrary(Rain):
    """Library created from observed rainfall data. inherits [`Rain`](./rain.html)."""

    def __init__(self,
                 rain: pandas.DataFrame,
                 time_step: pandas.Timedelta,
                 year_start: int,
                 block_size: pandas.Timedelta,
                 dry_year_break: float,
                 wet_year_break: float):
        """
        Args:
            rain: rain data of class `Rain`
            year_start: the month when the water year starts
            block_size: `pandas Timedelta` that counts as one block when sampling randomly for generation of
                synthetic rainfall data
            dry_year_break: water year with totals less than the value corresponding to this percentile will be
                regarded as dry year
            wet_year_break: water year with totals more than the value corresponding to this percentile will be
                regarded as wet year
        """

        Rain.__init__(self, rain, time_step)
        self.distribution_params = {}
        """parameters from fitting water year totals in observed data to theoretical distribution"""

        self._n_row = self.data.shape[0]

        self.year_start = year_start
        """the month when the water year begins"""

        self.block_size = pandas.to_timedelta(block_size)
        """the duration of short blocks used for randomly collating synthetic rainfall"""

        self._blocks = self.data.groupby(pandas.Grouper(freq=block_size)).ngroup()
        self.blocks = self._blocks.unique()
        """
        The unique block numbers for consecutive duration of short blocks used for randomly collating synthetic
        rainfall
        """

        year_breaks = self.totals.quantile([dry_year_break, wet_year_break]).round(2).tolist()
        self.dry_year_break = year_breaks[0]
        """water year with totals less than or equal to this value are dry-years"""

        self.wet_year_break = year_breaks[1]
        """water year with totals more than or equal to this value are wet-years"""

        self.year_groups = self._years_to_groups()
        """`dict` listing the dry, normal, and wet water years"""

        self._numpy_data = {}
        self._get_arr()

    def _get_arr(self):
        self._numpy_data["dry_years"] = self.data[self.year_groups["dry_years"]].to_numpy()
        self._numpy_data["normal_years"] = self.data[self.year_groups["normal_years"]].to_numpy()
        self._numpy_data["wet_years"] = self.data[self.year_groups["wet_years"]].to_numpy()

    def _years_to_groups(self):
        return {"dry_years": (self.totals[self.totals <= self.dry_year_break]).index.to_series(),
                "wet_years": (self.totals[self.totals >= self.wet_year_break]).index.to_series(),
                "normal_years": (self.totals[
                    (self.wet_year_break > self.totals) & (self.totals > self.dry_year_break)]).index.to_series()}

    def _fit_distribution(self, dist_names: Union[str, list[str]]):
        if isinstance(dist_names, str):
            dist_names = [dist_names]

        for dist_name in dist_names:
            self.distribution_params[dist_name] = _stats.fit_dist(self.totals, dist_name)

    def generate(self,
                 size: int,
                 dist: str = "gamma",
                 low: Union[pandas.Timedelta, str] = "6H",
                 high: Union[pandas.Timedelta, str] = "7D",
                 n_cores: int = 1,
                 n_digits: int = 2) -> SyntheticRain:
        """
        generate synthetic rainfall time-series data

        Args:
            size: the size (number of water-years) of randomly generated data
            dist: name of distribution to use. must be one of `gamma`, `gev`, or `lognorm`
            low: "duration of smallest block used for randomly collating synthetic data"
            high: "duration of largest block used for randomly collating synthetic data"
            n_cores: number of cores to use in parallel, if available
            n_digits: generated rainfall will be rounded to this many decimal places

        Returns:
            `Dataframe`: Synthetic rainfall time-series data for `size` number of water-years
        """

        n_cores = int(max(1, n_cores))
        low = pandas.Timedelta(low)
        high = pandas.Timedelta(high)

        assert _to_timestamp(low) > _to_timestamp(self.time_step), "'low' must be greater than time step"
        assert _to_timestamp(high) > _to_timestamp(low), "'high' must be larger than 'low'"

        if (self.distribution_params is None) or (dist not in self.distribution_params):
            self._fit_distribution(dist)

        params = self.distribution_params[dist]

        synthetic_totals = _stats.get_random_value(params, dist=dist, size=size).round(2)

        if n_cores > 1:
            pool = Pool(n_cores)
            container = []

            for synthetic_total in synthetic_totals:
                synthetic_rain_series = pool.apply_async(self._sample_rain,
                                                         (synthetic_total, n_digits, low, high))
                container.append(synthetic_rain_series)
            synthetic_rain = [res.get() for res in container]
        else:
            synthetic_rain = [self._sample_rain(val, n_digits, low, high) for val in synthetic_totals]

        synthetic_rain = pandas.concat(synthetic_rain, axis=1)
        synthetic_rain.index = self.data.index

        return SyntheticRain(data=synthetic_rain,
                             time_step=self.time_step,
                             block_size=self.block_size,
                             dist_name=dist,
                             dist_params=params,
                             low=low,
                             high=high)

    def _sample_rain(self,
                     synthetic_total: float,
                     n_digits: int,
                     low: pandas.Timedelta,
                     high: pandas.Timedelta) -> pandas.Series:
        """
        generate synthetic rainfall data

        :param synthetic_total: the total to assign to randomly concatenated rainfall time-series
        :param n_digits: generated rainfall will be rounded to this many decimal places
        :param low: "duration of smallest block used for randomly collating synthetic data"
        :param high: "duration of largest block used for randomly collating synthetic data"
        :return:
        """
        group = self._find_group(synthetic_total)
        low = int(low / self.time_step)
        high = int(high / self.time_step)
        random_inds = self._get_random_inds(low, high)
        n_cols = self._numpy_data[group].shape[1]
        row_inds, col_inds = self._get_row_col(random_inds, n_cols)

        collated_rain = pandas.Series(self._numpy_data[group][row_inds, col_inds])
        collated_rain_total = collated_rain.sum()
        synthetic_rain = collated_rain.mul(synthetic_total).div(collated_rain_total).round(n_digits)

        return synthetic_rain

    def _find_group(self, total):
        if total <= self.dry_year_break:
            return "dry_years"
        elif total >= self.wet_year_break:
            return "wet_years"
        return "normal_years"

    def _get_random_inds(self,
                         low: int,
                         high: int) -> numpy.array:
        max_size = math.ceil(self._n_row / low)
        inds = numpy.random.randint(low, high, max_size)
        cs_inds = inds.cumsum()
        inds = inds[cs_inds <= self._n_row]
        cs_inds = cs_inds[cs_inds <= self._n_row]
        if cs_inds[-1] < self._n_row:
            inds = numpy.append(inds, self._n_row - cs_inds[-1])

        return inds

    def _get_row_col(self,
                     inds: numpy.array,
                     n_cols: int):
        cols = numpy.random.randint(0, n_cols, len(inds))
        col_inds = numpy.repeat(cols, inds)
        row_inds = numpy.arange(self._n_row)
        return row_inds, col_inds
