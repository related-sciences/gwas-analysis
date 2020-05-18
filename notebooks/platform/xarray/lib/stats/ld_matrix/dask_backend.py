import numpy as np
import dask
import dask.dataframe as dd
from dask.dataframe import DataFrame
from xarray import Dataset, DataArray
from typing import Optional, Tuple
from ...dispatch import DaskBackend, dispatches_from
from ...dask_ext import dataframe_to_dataset
from .. import core
from . import cuda_backend


def get_ld_matrix_partitions(x, ais, cis, threshold, scores=None, index_dtype=np.int32, value_dtype=np.float32, **kwargs):
    # Chunk interval array contains information on how overlapping 
    # chunks should be defined
    cis = cis.to_dataset('var').to_dataframe()

    def to_ld_df(r, chunk_index):
        # Data must span before and after target indexes in block
        block_x = x[r['min_start']:r['max_stop']]
        # Intervals define target index span
        block_intervals = (
            ais[r['min_index']:r['max_index']],
            # 1D array with coords as strings: 'group' 'index' ... 'stop' 'count' 
            r.to_xarray()
        )
        # Scores must span same rows as data
        block_scores = scores[r['min_start']:r['max_stop']] if scores is not None else None

        f = dask.delayed(core._ld_matrix_block)(
            x=block_x,
            intervals=block_intervals,
            scores=block_scores,
            threshold=threshold, 
            return_value=True, 
            index_dtype=index_dtype, 
            value_dtype=value_dtype,
            **kwargs
        )
        meta = [
            ('i', index_dtype),
            ('j', index_dtype),
            ('value', value_dtype)
        ]
        if scores is not None:
            meta = meta.append(('cmp', np.int8))
        return dd.from_delayed([f], meta=meta)
    return {
        k: dd.concat([to_ld_df(r, i) for i, r in g.iterrows()])
        for k, g in cis.groupby('group')
    }

def get_ld_matrix_dataframe(partitions):
    return dd.concat([
        df.repartition(npartitions=1)
        for group, df in partitions.items()
    ])


@dispatches_from(core.ld_matrix)
def ld_matrix(
    ds: Dataset,
    intervals: Optional[Tuple[DataArray, DataArray]]=None,
    threshold: Optional[float]=None,
    scores=None,
    **kwargs
):
    """Dask backend for LD matrix computation
    
    See `core.ld_matrix` for primary documentation.
    """
    # This will coerce any dataset provided to the correct format
    # (if the coercion is supported by whatever dataset type was given)
    ds = ds.to.genotype_count_dataset()

    # 2D array (DataArray) of allele counts
    x = ds.data

    # Get partitions representing the result of each chunked computation, which is at least
    # broken up by `groups` if provided (possibly more if `target_chunk_size` set)
    ais, cis = intervals
    partitions = get_ld_matrix_partitions(x, ais, cis, threshold, scores=scores, **kwargs)

    # Concatenate partitions noting that each partition represents ALL data
    # for any one group (e.g. chromosome)
    ldm = get_ld_matrix_dataframe(partitions)

    return ldm

core.register_backend_function(DaskBackend)(ld_matrix)
