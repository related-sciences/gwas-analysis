from xarray import Dataset
import pandas as pd
import dask.dataframe as dd
from ...typing import DataMapping
from ...dispatch import dispatches_from, DaskBackend
from .. import core
from . import numba_backend


@dispatches_from(core.maximal_independent_set)
def maximal_independent_set(
    ds: DataMapping,
    **kwargs
) -> DataMapping:
    """Dask MIS

    Returns
    -------
    dask.dataframe.DataFrame
    """
    df = ds
    if isinstance(df, Dataset):
        df = df.to_dataframe()
    if isinstance(df, pd.DataFrame):
        df = dd.from_pandas(df, npartitions=1)
    if not isinstance(df, dd.DataFrame):
        raise ValueError(f'Unable to coerce mapping of type "{type(df)}" to dask DataFrame')
    def func(df):
        ds = numba_backend.maximal_independent_set(df, **kwargs)
        return ds.to_dataframe()
    return dd.map_partitions(func, df, meta=[('index_to_drop', df.dtypes['i'])])

core.register_backend_function(DaskBackend)(maximal_independent_set)
