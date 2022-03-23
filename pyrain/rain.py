import pandas
import pdrle
from scipy import interpolate
from typing import Union
from ._utils import _to_timestamp


class Rain:
    """
    class that holds rainfall data in wide format and provides useful attributes and methods for viewing results. this
    class is inherited by [`RainLibrary`](./rain_library.html) and [`SyntheticRain`](./synthetic_rain.html).
    """

    def __init__(self,
                 data: pandas.DataFrame,
                 time_step: pandas.Timedelta):
        self.data = data
        """
        `DataFrame` containing rainfall data in wide format. columns are different water years. index is `DatetimeIndex`
        that goes from 2018 to 2019.
        """

        self.time_step = time_step
        """time step of the rainfall data."""

        self.totals = self.data.sum()
        """the total rainfall in each water-year"""

    def get_exceedance(self,
                       windows: list[Union[pandas.Timedelta, str]],
                       n_digits: int = 2,
                       probs: list[float] = None) -> Union[dict[pandas.DataFrame], pandas.DataFrame]:
        """
        for annual maxima values of rainfall depths of durations in `windows`, calculate probability of being equalled
        or exceeded.

        Args:
            windows: time duration over which rainfall values will be totalled
            n_digits: rainfall values will be rounded to this many decimal places before calculating totals
            probs: optional parameter indicating the exceedance probabilities for which corresponding rainfall
                values will be interpolated and returned.

        Returns:
            if `probs` is provided, a `DataFrame` where the `index` is `probs` and columns are `windows`. if `probs`
                is empty, If empty, a `dict` of `DataFrame` where the keys are `probs`.
        """
        arr = {}

        if isinstance(windows, str):
            windows = [windows]

        for window in windows:
            assert _to_timestamp(window) >= _to_timestamp(
                self.time_step), "windows need to be larger than or equal to time step"

        for window in windows:
            window = pandas.Timedelta(window)
            nm = f'{window.total_seconds() / 3600}H'
            max_: pandas.Series = self.data.rolling(window=window).sum().max().round(n_digits)
            ex = max_.value_counts().reset_index()
            ex.columns = ["val", "len"]
            ex = ex.sort_values("val", ascending=False)
            ex.len = ex.len.cumsum()
            ex["prob"] = ex.len.div(len(max_) + 1)

            if probs is not None:
                interpolate_vals = interpolate.interp1d(ex.prob, ex.val, bounds_error=False)
                arr[nm] = interpolate_vals(probs)
            else:
                arr[nm] = ex[["val", "prob"]]

        return pandas.DataFrame(arr, index=probs) if probs is not None else arr

    def get_wet_period_count(self,
                             period: str,
                             group_by: str = None) -> pandas.DataFrame:
        """
        calculate number of wet periods in each water year

        Args:
            period: the duration of wet-periods to count - `offset alias`
            group_by: the duration by which the period counts should be grouped by - `offset alias`

        Returns:
            `Dataframe`: counts of `period` duration with rainfall by `group_by` duration
        """

        assert _to_timestamp(period) >= _to_timestamp(self.time_step), "period must be larger than time step"

        wp = self.data.groupby(pandas.Grouper(freq=period)).sum().gt(0)

        if group_by is None:
            return wp.sum()

        assert _to_timestamp(group_by) > _to_timestamp(period), "group_by cannot be smaller than duration"
        return wp.groupby(pandas.Grouper(freq=group_by)).sum()

    def get_summary(self,
                    funcs: Union[str, list[str]] = "sum",
                    group_by: str = None):
        """

        Args:
            funcs: the functions to use for summarizing. default is `sum`. can also be a list of functions.
            group_by: the duration by which the summary should be grouped by - `offset alias`

        Returns:
            `Dataframe`: summary data
        """

        if group_by is None:
            return self.data.agg(funcs)

        assert _to_timestamp(group_by) >= _to_timestamp(self.time_step), "group_by must be larger than time step"
        return self.data.groupby(pandas.Grouper(freq=group_by)).agg(funcs)

    def get_storm_durations(self,
                            group_by: str = None):
        pass
