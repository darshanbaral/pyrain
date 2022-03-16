from multiprocessing import Pool
from . import _stats
from .rain import Rain
from .synthetic_rain import SyntheticRain
import pandas
from typing import Union


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
                 n_cores: int = 1,
                 n_digits: int = 2) -> SyntheticRain:
        """
        generate synthetic rainfall time-series data

        Args:
            size: the size (number of water-years) of randomly generated data
            dist: name of distribution to use. must be one of `gamma`, `gev`, or `lognorm`
            n_cores: number of cores to use in parallel, if available
            n_digits: generated rainfall will be rounded to this many decimal places

        Returns:
            `Dataframe`: Synthetic rainfall time-series data for `size` number of water-years
        """

        n_cores = int(max(1, n_cores))

        if (self.distribution_params is None) or (dist not in self.distribution_params):
            self._fit_distribution(dist)

        params = self.distribution_params[dist]

        synthetic_totals = _stats.get_random_value(params, dist=dist, size=size).round(2)

        if n_cores > 1:
            pool = Pool(n_cores)
            container = []

            for synthetic_total in synthetic_totals:
                synthetic_rain_series = pool.apply_async(self._sample_rain, (synthetic_total, n_digits))
                container.append(synthetic_rain_series)
            synthetic_rain = [res.get() for res in container]
        else:
            synthetic_rain = [self._sample_rain(val, n_digits) for val in synthetic_totals]

        synthetic_rain = pandas.concat(synthetic_rain, axis=1)
        synthetic_rain.index = self.data.index

        return SyntheticRain(data=synthetic_rain,
                             time_step=self.time_step,
                             block_size=self.block_size,
                             dist_name=dist,
                             dist_params=params)

    def _sample_rain(self,
                     synthetic_total: float,
                     n_digits: int) -> pandas.Series:
        """
        generate synthetic rainfall data

        :param synthetic_total: the total to assign to randomly concatenated rainfall time-series
        :param n_digits: generated rainfall will be rounded to this many decimal places
        :return:
        """
        group = self._find_group(synthetic_total)

        curr_year_rain = {}
        for block in self.blocks:
            block_year = self.year_groups[group].sample(1).item()  # get random year
            block_dates = self.data.index[self._blocks == block]
            curr_year_rain[block] = self.data.loc[block_dates, block_year]
        curr_year_rain = pandas.concat(curr_year_rain, axis=0)
        collated_sum = curr_year_rain.sum()
        curr_year_rain = curr_year_rain.mul(synthetic_total).div(collated_sum)

        return curr_year_rain.round(n_digits)

    def _find_group(self, total):
        if total <= self.dry_year_break:
            return "dry_years"
        elif total >= self.wet_year_break:
            return "wet_years"
        return "normal_years"
