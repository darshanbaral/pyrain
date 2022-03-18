"""
# pyrain

A python package for rainfall simulation - stochastically generating synthetic rainfall time series data from
observed data. Made by [Darshan Baral](https://www.darshanbaral.com/)

## Method

Synthetic rainfall data is generated from observed rainfall time-series data.

1. The observed rainfall time-series data is separated into different water-years
2. Total rainfall in each water-year is calculated to obtain *observed totals*.
3. Classify *observed water-years* into buckets of `dry year`, `wet year`, and `normal year` based on pre-defined
    percentile threshold in the *observed totals*. By default, the water years where the observed total is less
    than or equal to the 25th percentile are classified as dry years. Similarly, the water years where the total
    rainfall is greater than or equal to the 75th percentile are classified as wet years. The remaining years are
    classified as normal years.
4. Generate *synthetic totals*:
    - Fit a theoretical distribution to the *observed totals*. Can choose from `gamma`, `gev`, or `lognorm`.
    - Generate the required number of *synthetic totals* from the theoretical distribution.
5. For each *synthetic total* value from Step `4`,
    - Determine whether the *synthetic total* would be classified as `dry year`, `wet year`, or `normal year` based on
        the threshold determined from the observed data (Step `4`).
    - Pick blocks of short durations one at a time from different water-years randomly in the bucket determined
        above to collate a synthetic time series. By default, blocks of durations between 6 hours and 7 days are used. For example, if
        the synthetic total is determined to belong to dry year bucket, first randomly chosen n-days (6H < n < 7D)would come from a random
        observed water-year in dry bucket. Then the next randomly chosen m-days (6H < m < 7D) would again come from a random water-year chosen
        from the dry bucket. Blocks would keep stacking one at a time from random water-years of dry bucket until
        we had a time series for the whole synthetic water-year.
    - Normalize the time-series from above by its total and multiply by the *synthetic total*.

## Installation

1. Download the latest distribution file [`pyrain-x.x.x.tar.gz`](https://github.com/darshanbaral/pyrain/releases).
2. Install the tar file with `pip install <tar file>`.

## Usage

```python
import pyrain

rain = pyrain.Rain("./data/hourly.csv", "datetime", "P")
lib = rain.create_library()
synthetic_rain = lib.generate(100, n_cores=4)
```

## Links

- [GitHub](https://github.com/darshanbaral/pyrain)
"""

from .observed_rain import ObservedRain
from .rain_library import RainLibrary, load_synthetic_rain
