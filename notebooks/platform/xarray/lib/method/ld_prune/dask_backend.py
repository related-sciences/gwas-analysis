from xarray import Dataset
from typing import Optional
import dask.dataframe as dd
from ...dispatch import DaskBackend, dispatches_from
from ...graph.maximal_independent_set import numba_backend
from ...typing import DataFrame
from .. import core

 
@dispatches_from(core.ld_prune)
def ld_prune(
    ldm: DataFrame,
    use_cmp: bool = True,
    **kwargs
) -> DataFrame:
    """LD Prune (Dask)

    See `method.core.ld_prune` for documentation.
    """
    if not use_cmp and 'cmp' in ldm:
        ldm = ldm.drop(['cmp'], axis=1)
    def func(df):
        ds = numba_backend.maximal_independent_set(df)
        return ds.to_dataframe()
    return dd.map_partitions(func, ldm, meta=[('index_to_drop', ldm.dtypes['i'])])

core.register_backend_function(DaskBackend)(ld_prune)
