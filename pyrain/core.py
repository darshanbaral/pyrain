import warnings
import pandas
from pandas.api import types
from multiprocessing import Pool
from datetime import datetime
from typing import Union
from . import stats


class Rain:
    def __init__(self,
                 path: str,
                 datetime_col: str,
                 rain_col: str,
                 date_format: str = None):
        """
        Read rain data and prepare for library creation.

        :param path: path to the csv file containing rain data
        :param datetime_col: name of the datetime column in the csv
        :param rain_col: name of the rain column in the csv
        :param date_format: format to use for parsing datetime column
        """
        self.rain_data = self.__read_rain(path, datetime_col, rain_col, date_format)

    def __read_rain(self, path, datetime_col, rain_col, date_format):
        rain_data = pandas.read_csv(path, usecols=[datetime_col, rain_col])
        rain_data.columns = ["datetime", "P"]
        rain_data.datetime = pandas.to_datetime(rain_data.datetime, format=date_format)
        rain_data = rain_data.set_index("datetime")

        if not types.is_numeric_dtype(rain_data.P):
            rain_data.P = pandas.to_numeric(rain_data.P, errors='coerce')

        if rain_data.P.isna().any():
            warnings.warn("NA values in rainfall were filled with zero")
            rain_data = rain_data.fillna(0)

        time_steps = rain_data.index.to_series().diff().dropna()
        self.time_step = time_steps.mode().item()

        if time_steps.unique().shape[0] > 1:
            time_step_counts = time_steps.value_counts().reset_index().head(12)
            time_step_counts.columns = ["time_step", "counts"]
            warnings.warn("\nGaps were found in data and were filled with zero")
            warnings.warn("\n" + time_step_counts.to_string())
            rain_data = rain_data.reindex(pandas.date_range(rain_data.index.min(),
                                                            rain_data.index.max(),
                                                            freq=self.time_step),
                                          fill_value=0)
        return rain_data


class SyntheticRain:
    def __init__(self,
                 data: pandas.DataFrame,
                 time_step: pandas.Timedelta,
                 dist_name: str,
                 dist_params: tuple[float]):
        self.data = data
        self.time_step = time_step
        self.dist_name = dist_name
        self.dist_params = dist_params


class RainLibrary:
    def __init__(self,
                 rain: Rain,
                 year_start: int = 9,
                 block_size: Union[str, pandas.Timedelta] = "3D",
                 dry_year_break: float = 0.25,
                 wet_year_break: float = 0.75):
        """
        Library created from observed rainfall data.

        :param rain: rain data of class Rain
        :param year_start: the month when the water year starts
        :param block_size: Timedelta that counts as one block when sampling randomly for generation of
        synthetic rainfall data
        :param dry_year_break: water year with totals less than the value corresponding to this percentile will be
        regarded as dry year
        :param wet_year_break: water year with totals more than the value corresponding to this percentile will be
        regarded as wet year
        """

        assert 1 > dry_year_break > 0, "'dry_year_break' should be between 0 and 1"
        assert 1 > wet_year_break > 0, "'wet_year_break' should be between 0 and 1"
        assert wet_year_break > dry_year_break, "'wet_year_break' should be greater than 'dry_year_break'"

        self.distribution_params = {}
        self.time_step = rain.time_step
        self.year_start = year_start
        self.block_size = pandas.to_timedelta(block_size)
        self.group_column = "water_year"
        self.table, self.blocks, self.index = self._create_library(rain.rain_data)
        self.totals = self.table.sum()
        year_breaks = self.totals.quantile([dry_year_break, wet_year_break]).round(2)
        self.dry_year_break, self.wet_year_break = year_breaks.tolist()
        self.year_groups = self._years_to_groups()

    def _create_library(self, rain: pandas.DataFrame):
        rain[self.group_column] = rain.index.map(
            lambda x: x.year if (x.month < self.year_start) | (self.year_start == 1) else x.year + 1)

        # Remove leap day
        rain = rain[~((rain.index.month == 2) & (rain.index.day == 29))]

        rain.index = rain.index.strftime("%m%d%H%M")
        rain = rain.pivot(columns=self.group_column, values="P")

        # Remove years with only partial data
        is_partial_year = rain.isna().any()
        rain = rain.loc[:, ~is_partial_year]

        new_index = pandas.date_range(datetime(2018, self.year_start, 1),
                                      datetime(2019, self.year_start, 1) - self.time_step,
                                      freq=self.time_step)
        block_num = new_index.to_series().groupby(pandas.Grouper(freq="3D")).ngroup()
        rain.index = block_num
        return rain, set(block_num), new_index

    def _years_to_groups(self):
        return {"dry_years": (self.totals[self.totals <= self.dry_year_break]).index.to_series(),
                "wet_years": (self.totals[self.totals >= self.wet_year_break]).index.to_series(),
                "normal_years": (self.totals[
                    (self.wet_year_break > self.totals) & (self.totals > self.dry_year_break)]).index.to_series()}

    def _fit_distribution(self, dist_names: Union[str, list[str]]):
        """
        fit distributions to water year totals

        :param dist_names: name or names of distributions
        :return: distribution parameters fitted to water year totals
        """
        if isinstance(dist_names, str):
            dist_names = [dist_names]

        for dist_name in dist_names:
            self.distribution_params[dist_name] = stats.fit_dist(self.totals, dist_name)

    def generate(self, size: int, dist: str = "gamma", n_cores: int = 1) -> SyntheticRain:
        """
        generate random values from distribution fitted to water year totals

        :param size: size of randomly generated data
        :param dist: name of distribution to use
        :param n_cores: number of cores to use in parallel, if available
        :return: randomly generated values from fitted distribution
        """

        n_cores = int(max(1, n_cores))

        if (self.distribution_params is None) or (dist not in self.distribution_params):
            self._fit_distribution(dist)

        params = self.distribution_params[dist]

        synthetic_totals = stats.get_random_value(params, dist=dist, size=size).round(2)

        if n_cores > 1:
            pool = Pool(n_cores)
            container = []

            for synthetic_total in synthetic_totals:
                synthetic_rain_series = pool.apply_async(self._sample_rain, (synthetic_total,))
                container.append(synthetic_rain_series)
            synthetic_rain = [res.get() for res in container]
        else:
            synthetic_rain = [self._sample_rain(val) for val in synthetic_totals]

        synthetic_rain = pandas.concat(synthetic_rain, axis=1)
        synthetic_rain.index = self.index

        return SyntheticRain(synthetic_rain,
                             self.time_step,
                             dist,
                             params)

    def _sample_rain(self, synthetic_total: float) -> pandas.Series:
        """
        generate synthetic rainfall data

        :param synthetic_total: the total to assign to randomly concatenated rainfall time-series
        :return:
        """
        group = self._find_group(synthetic_total)

        curr_year_rain = {}
        for block in self.blocks:
            block_year = self.year_groups[group].sample(1).item()  # get random year
            curr_year_rain[block] = self.table.loc[block, block_year]
        curr_year_rain = pandas.concat(curr_year_rain, axis=0)
        curr_year_rain = curr_year_rain.mul(synthetic_total) / curr_year_rain.sum()

        return curr_year_rain

    def _find_group(self, total):
        if total <= self.dry_year_break:
            return "dry_years"
        elif total >= self.wet_year_break:
            return "wet_years"
        return "normal_years"
