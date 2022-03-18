import pandas
from typing import Union
from pathlib import Path
from .rain import Rain


class SyntheticRain(Rain):
    """Class that holds synthetic rainfall data. inherits [`Rain`](./rain.html)"""
    def __init__(self,
                 data: pandas.DataFrame,
                 time_step: pandas.Timedelta,
                 block_size: pandas.Timedelta,
                 dist_name: str,
                 dist_params: tuple[float],
                 low: pandas.Timedelta,
                 high: pandas.Timedelta):

        Rain.__init__(self, data, time_step)
        self.block_size = block_size
        """block size duration used for collating synthetic data - `Timedelta`"""

        self.dist_name = dist_name
        """name of distribution used for generating synthetic water-year totals"""

        self.dist_params = dist_params
        """parameters from fitting `dist_name` distribution to observed water-year totals"""

        self.low = low
        """Duration of smallest block used for randomly collating synthetic data"""

        self.high = high
        """Duration of largest block used for randomly collating synthetic data"""

    def save(self,
             root: Union[str, Path],
             prefix: str = "",
             save_info: bool = True,
             n_digits: int = 2):
        """
        Save synthetic rainfall data locally

        Args:
            root: the directory where the synthetic rain data should be saved
            prefix: the prefix that will be added to the file names
            save_info: if `False`, only the rain data will be saved
            n_digits: the rainfall values will be rounded to this many decimal places
        """
        root = Path(root)
        self.data.round(n_digits).to_csv(root / "{prefix}_synthetic_rain.csv".format(prefix=prefix))
        if save_info:
            with open(root / "{prefix}_synthetic_rain_info.toml".format(prefix=prefix), "w") as f:
                f.write("'time_step' = '{ts}'\n".format(ts=self.time_step))
                f.write("'block_size' = '{bs}'\n".format(bs=self.block_size))
                f.write("'dist_name' = '{dn}'\n".format(dn=self.dist_name))
                f.write("'dist_params' = {dp}\n".format(dp=list(self.dist_params)))
                f.write("'low' = '{low}'\n".format(low=self.low))
                f.write("'high' = '{high}'\n".format(high=self.high))
            f.close()
