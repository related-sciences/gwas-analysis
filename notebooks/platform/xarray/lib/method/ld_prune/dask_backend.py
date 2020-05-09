from xarray import Dataset
from typing import Optional
import dask.dataframe as dd
from dask.dataframe import DataFrame
from ...dispatch import ClassBackend, register_backend
from ...graph.core import maximal_independent_set
from ..core import DOMAIN

 
@register_backend(DOMAIN)
class DaskBackend(ClassBackend):

    id = 'dask'

    def ld_prune(
        self,
        ldm: DataFrame,
        use_cmp: bool = True
    ):
        """LD Prune (Dask)

        See `method.core.ld_prune` for documentation.
        """
        df = dd.map_partitions(maximal_independent_set, ldm, meta=[('index_to_drop', ldm.dtypes['i'])])
        return df['index_to_drop'].values






