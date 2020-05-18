from typing import Union, Mapping
from pathlib import Path
import pandas as pd
import dask.dataframe as dd
import dask.array as da
import xarray as xr
import numpy as np

PathType = Union[str, Path]

# TODO: How can a union type be specified with optional dependencies (i.e. Dask)?
DataFrame = Union[pd.DataFrame, dd.DataFrame]
DataMapping = Union[xr.Dataset, DataFrame]
BlockArray = Union[xr.DataArray, np.ndarray]