# pyrain

A python package for stochastically simulating rainfall time series data from observed data.

## Installation

1. Download the latest distribution file [`pyrain-x.x.x.tar.gz`](https://github.com/darshanbaral/pyrain/releases).
2. Install the tar file with `pip install <tar file>`.

## Usage

```python
import pyrain

coe = pyrain.Rain("./data/hourly.csv", "datetime", "P", "%Y-%m-%d %H:%M:%S")

d = pyrain.RainLibrary(coe)
r = d.generate(5)
```
