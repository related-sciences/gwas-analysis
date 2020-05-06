from . import cuda_backend
from ..interval import axis_intervals
import numpy as np


def get_ld_matrix_partitions(x, ais, cis, threshold, scores=None, index_dtype=np.int32, value_dtype=np.float32):
    import dask
    import dask.dataframe as dd

    df = cis.to_dataset('var').to_dataframe()
    def to_ld_df(r, chunk_index):
        arrays = dict(
            x=x[r['min_start']:r['max_stop']],
            axis_intervals=ais.data[r['min_index']:r['max_index']],
            chunk_interval=r.values.reshape((1, -1))
        )
        if scores is not None:
            arrays['scores'] = scores[r['min_start']:r['max_stop']]
        # TODO: Make this call through a frontend dispatcher for the stats package instead
        f = dask.delayed(cuda_backend.ld_matrix)(
            **arrays, 
            min_threshold=threshold, 
            return_value=True, 
            index_dtype=index_dtype, 
            value_dtype=value_dtype,
            lock=False
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
        for k, g in df.groupby('group')
    }

def get_ld_matrix_dataframe(partitions):
    import dask.dataframe as dd
    return dd.concat([
        df.repartition(npartitions=1)
        for group, df in partitions.items()
    ])

def ld_matrix(x, window, threshold=None, step=None, groups=None, positions=None, scores=None, target_chunk_size=None, return_intervals=False):
    n = x.shape[0]

    # Compute row intervals as well as any necessary overlap
    ais, cis = axis_intervals(n=n, window=window, step=step, groups=groups, positions=positions, target_chunk_size=target_chunk_size) 

    # Get partitions representing the result of each chunked computation, which is at least
    # broken up by `groups` if provided (possibly more if `target_chunk_size` set)
    partitions = get_ld_matrix_partitions(x, ais, cis, threshold, scores=scores)

    # Concatenate partitions noting that each partition represents ALL data
    # for any one group (e.g. chromosome)
    ldm = get_ld_matrix_dataframe(partitions)

    if return_intervals:
        return ldm, (ais, cis)
    else:
        return ldm