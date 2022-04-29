import numpy
import pandas
import pdrle
from scipy import interpolate
from typing import Union, Callable
from ._utils import _to_timestamp, _calc_exceedance


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
            max_ = self.data.rolling(window=window).sum().max().round(n_digits)
            ex = _calc_exceedance(max_)

            if probs is not None:
                interpolate_vals = interpolate.interp1d(ex.prob, ex.val, bounds_error=False)
                arr[nm] = interpolate_vals(probs)
            else:
                arr[nm] = ex[["val", "prob"]]

        return pandas.DataFrame(arr, index=probs) if probs is not None else arr

    def get_total_wet_periods(self,
                              period: str,
                              group_by: str = None) -> pandas.DataFrame:
        """
        *calculate total number of wet periods in each water year - e.g. total wet hours ech year*

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
                    funcs: Union[str, list[str], Callable, list[Callable]] = "sum",
                    group_by: str = None):
        """
        summarize rainfall data for each water

        Args:
            funcs: the functions to use for summarizing. default is `sum`. can also be a list of functions.
            group_by: the duration by which the summary should be grouped by - `offset alias`

        Returns:
            `Dataframe`: summary data
        """

        if not isinstance(funcs, list):
            funcs = [funcs]

        if group_by is None:
            return self.data.agg(funcs)

        assert _to_timestamp(group_by) >= _to_timestamp(self.time_step), "group_by must be larger than time step"
        return self.data.groupby(pandas.Grouper(freq=group_by)).agg(funcs)

    def get_storm_duration_summary(self,
                                   funcs,
                                   group_by=None,
                                   min_dur=None,
                                   ignore_dur="1H") -> pandas.DataFrame:
        """
        *summarize rainfall data for each water year by storm durations*

        Args:
            funcs: the functions to use for summarizing. default is `sum`. can also be a list of functions.
            group_by: the duration by which the summary should be grouped by - `offset alias`
            min_dur: the minimum duration of incremental periods to use for summarizing data
                default is 4 times the time step
            ignore_dur: dry periods smaller than this duration will be considered part of wet period

        Returns:
            `pandas.DataFrame`: summary by duration of storms
        """

        if min_dur is None:
            min_dur = self.time_step * 4
        else:
            assert _to_timestamp(min_dur) >= _to_timestamp(self.time_step),\
                "min_dur must be larger than time step"

        if group_by is not None:
            assert _to_timestamp(group_by) >= _to_timestamp(min_dur),\
                "group_by must be larger than time step and min_dur"

        if not isinstance(funcs, list):
            funcs = [funcs]

        ans = pandas.concat([self._summary_by_storms(self.data[nm],
                                                     funcs,
                                                     group_by,
                                                     min_dur,
                                                     ignore_dur) for nm in self.data.columns],
                            axis=1)
        return ans.sort_index()

    @staticmethod
    def _summary_by_storms(data: pandas.Series,
                           funcs: Union[list[str, Callable]],
                           group_by: str,
                           min_dur: Union[str, pandas.DateOffset],
                           ignore_dur: Union[str, pandas.Timedelta]):
        """
        :param data: data for one water-year
        :param funcs: the functions to use for summarizing. default is `sum`. can also be a list of functions.
        :param group_by: the duration by which the summary should be grouped by - `offset alias`
        :param min_dur: the minimum duration of incremental periods to use for summarizing data.
            default is 4 times the time step.
        :param ignore_dur: dry periods smaller than this duration will be considered part of wet period
        """

        event_id = pdrle.get_id(data.eq(0))
        if not pandas.isnull(ignore_dur):
            ignore_dur = pandas.Timedelta(ignore_dur)
            events = pandas.concat([data.index.to_series().groupby(event_id).agg([numpy.ptp, "count"]).reset_index(),
                                    data.groupby(event_id).sum().eq(0).rename("is_dry")], axis=1)
            events.loc[events.ptp.le(ignore_dur), "is_dry"] = False
            event_id = events.is_dry.repeat(events["count"])
            event_id.index = data.index
            event_id = pdrle.get_id(event_id)

        durations = data.index.to_series().dt.ceil(min_dur).groupby(event_id).transform(numpy.ptp)
        durations.loc[durations.lt(ignore_dur)] = ignore_dur

        if group_by is None:
            grouping_data = [event_id, durations]
            prelim_summary = data.groupby(grouping_data).agg(sum)
            prelim_summary.index.names = ["event_id", "durations"]
            group_id = "durations"
        else:
            grouping_data = [pandas.Grouper(freq=group_by), event_id, durations]
            prelim_summary = data.groupby(grouping_data).agg(sum)
            prelim_summary.index.names = ["group_by", "event_id", "durations"]
            group_id = ["group_by", "durations"]

        prelim_summary = prelim_summary.loc[prelim_summary.gt(0)]
        summary: pandas.DataFrame = prelim_summary.groupby(group_id).agg(funcs)
        summary.columns = pandas.MultiIndex.from_tuples([(data.name, nm) for nm in summary.columns])
        return summary
