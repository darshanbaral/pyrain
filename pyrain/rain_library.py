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
        time_step, dist_name, dist_params, low, high = [None] * 5
    else:
        info = toml.load(info_path)
        time_step = pandas.Timedelta(info["time_step"])
        dist_name = info["dist_name"]
        low = pandas.Timedelta(info["low"])
        high = pandas.Timedelta(info["high"])
        dist_params = tuple(info["dist_params"])

    return SyntheticRain(data, time_step, dist_name, dist_params, low, high)


class RainLibrary(Rain):
    """Library created from observed rainfall data. inherits [`Rain`](./rain.html)."""

    def __init__(self,
                 rain: pandas.DataFrame,
                 time_step: pandas.Timedelta,
                 year_start: int,
                 dry_year_break: float,
                 wet_year_break: float):
        """
        Args:
            rain: rain data of class `Rain`
            year_start: the month when the water year starts
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
                 n_digits: int = 2) -> SyntheticRain:
        """
        generate synthetic rainfall time-series data

        Args:
            size: the size (number of water-years) of randomly generated data
            dist: name of distribution to use. must be one of `gamma`, `gev`, or `lognorm`
            low: "duration of smallest block used for randomly collating synthetic data"
            high: "duration of largest block used for randomly collating synthetic data"
            n_digits: generated rainfall will be rounded to this many decimal places

        Returns:
            `Dataframe`: Synthetic rainfall time-series data for `size` number of water-years
        """

        low = pandas.Timedelta(low)
        high = pandas.Timedelta(high)

        assert _to_timestamp(low) > _to_timestamp(self.time_step), "'low' must be greater than time step"
        assert _to_timestamp(high) > _to_timestamp(low), "'high' must be larger than 'low'"

        n_rows_low = int(low / self.time_step)
        n_rows_high = int(high / self.time_step)

        if (self.distribution_params is None) or (dist not in self.distribution_params):
            self._fit_distribution(dist)

        params = self.distribution_params[dist]

        synthetic_totals = numpy.sort(_stats.get_random_value(params, dist=dist, size=size).round(2))
        dry_year_ind = synthetic_totals.searchsorted(self.dry_year_break, side="right")
        wet_year_ind = synthetic_totals.searchsorted(self.wet_year_break, side="left")
        synthetic_total_buckets = {"dry_years": synthetic_totals[:dry_year_ind],
                                   "normal_years": synthetic_totals[dry_year_ind:wet_year_ind],
                                   "wet_years": synthetic_totals[wet_year_ind:]}

        synthetic_rain = []
        for bucket_name, totals in synthetic_total_buckets.items():
            bucket = self.data[self.year_groups[bucket_name]].to_numpy()
            _, n_cols = bucket.shape
            inds = [self._get_row_col(self._get_random_inds(n_rows_low, n_rows_high), n_cols=n_cols) for _ in totals]
            bucket_rain = numpy.column_stack([bucket[row_ind, col_ind] for row_ind, col_ind in inds])
            bucket_rain = bucket_rain * totals / bucket_rain.sum(axis=0)
            bucket_rain = pandas.DataFrame(bucket_rain.round(n_digits))
            bucket_rain.index = self.data.index
            synthetic_rain.append(bucket_rain)

        synthetic_rain = pandas.concat(synthetic_rain, axis=1)
        synthetic_rain.columns = range(len(synthetic_rain.columns))

        return SyntheticRain(data=synthetic_rain,
                             time_step=self.time_step,
                             dist_name=dist,
                             dist_params=params,
                             low=low,
                             high=high)

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
