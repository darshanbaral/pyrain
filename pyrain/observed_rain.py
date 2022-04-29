import warnings
import pandas
from pandas.api import types
from datetime import datetime
from typing import Union
from .rain_library import RainLibrary
from ._utils import _get_water_year


class ObservedRain:
    def __init__(self,
                 rain: Union[str, pandas.DataFrame],
                 datetime_col: str,
                 rain_col: str,
                 date_format: str = None):
        """
        Class that facilitates reading raw rainfall data and preparing for library creation.

        Args:
            rain: pandas Dataframe with rain data or path to the csv file containing rain data
            datetime_col: name of the datetime column in the csv
            rain_col: name of the rain column in the csv
            date_format: format to use for parsing datetime column
        """
        self.time_step = None
        "time step calculated from raw data"

        self.data = self.__read_rain(rain, datetime_col, rain_col, date_format)
        "processed rainfall data in long format"

    def __read_rain(self, data, datetime_col, rain_col, date_format):
        if isinstance(data, str):
            rain_data = pandas.read_csv(data, usecols=[datetime_col, rain_col])
        elif isinstance(data, pandas.DataFrame):
            rain_data = data[[datetime_col, rain_col]].copy()
        else:
            raise ValueError("'data' is not valid")

        rain_data.columns = ["datetime", "P"]

        if not types.is_datetime64_any_dtype(rain_data["datetime"]):
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
            warnings.warn("\nGaps were found in data")
            warnings.warn("\n" + time_step_counts.to_string())
            rain_data = rain_data.reindex(pandas.date_range(rain_data.index.min(),
                                                            rain_data.index.max(),
                                                            freq=self.time_step))
        return rain_data

    def create_library(self,
                       year_start: int = 9,
                       dry_year_break: float = 0.25,
                       wet_year_break: float = 0.75) -> RainLibrary:
        """
        create a library o facilitate synthetic rainfall generation. converts rainfall data in long format to wide
        format where each column is different water year. The water years where data is not available for all twelve
        months are dropped. In wide format, all water years will have same datetime index. dates will go from year
        2017 to 2018. Data for leap days will be dropped to keep shape uniform in wide format

        Args:
            year_start: the month when the water year starts. default is `9` (for September)
            dry_year_break: water year with totals less than the value corresponding to this percentile will be
                regarded as dry year
            wet_year_break: water year with totals more than the value corresponding to this percentile will be
                regarded as wet year

        Returns:
            object of class [`RainLibrary`](./rain_library.html)
        """

        assert 1 > dry_year_break > 0, "'dry_year_break' should be between 0 and 1"
        assert 1 > wet_year_break > 0, "'wet_year_break' should be between 0 and 1"
        assert wet_year_break > dry_year_break, "'wet_year_break' should be greater than 'dry_year_break'"
        assert isinstance(self.data.index, pandas.DatetimeIndex), "index of rain data should be a datetime index"

        group_column = "water_year"
        rain = self.data.copy()
        rain[group_column] = _get_water_year(rain.index, year_start)
        rain = rain[~((rain.index.month == 2) & (rain.index.day == 29))]  # Remove leap day

        rain.index = rain.index.strftime("%m%d%H%M")
        rain = rain.pivot(columns=group_column, values="P")
        rain.columns.name = None

        for _, row in rain.iterrows():
            self._impute(row)

        new_index = pandas.date_range(datetime(2018, year_start, 1),
                                      datetime(2019, year_start, 1) - self.time_step,
                                      freq=self.time_step)
        rain.index = new_index
        return RainLibrary(rain, self.time_step, year_start, dry_year_break, wet_year_break)

    @staticmethod
    def _impute(data: pandas.Series) -> None:
        """

        :param data:
        :return:
        """

        if (data.notna().sum() >= 5) and (data.notna().sum() >= (data.size // 3)):
            data[data.isna()] = data.median()
        else:
            data[data.isna()] = 0
