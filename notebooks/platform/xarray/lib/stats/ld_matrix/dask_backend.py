import numpy as np
import dask
import dask.dataframe as dd
from dask.dataframe import DataFrame
from xarray import Dataset
from typing import Optional
from ...dispatch import ClassBackend, register_backend
from ..core import DOMAIN
from ..interval import axis_intervals
from . import cuda_backend


def get_ld_matrix_partitions(x, ais, cis, threshold, scores=None, index_dtype=np.int32, value_dtype=np.float32, **kwargs):

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
        # (this is one place where a user should potentially have control over what happens)
        f = dask.delayed(cuda_backend.ld_matrix)(
            **arrays, 
            min_threshold=threshold, 
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
        for k, g in df.groupby('group')
    }

def get_ld_matrix_dataframe(partitions):
    import dask.dataframe as dd
    return dd.concat([
        df.repartition(npartitions=1)
        for group, df in partitions.items()
    ])


@register_backend(DOMAIN)
class DaskBackend(ClassBackend):

    id = 'dask'

    def ld_matrix(
        self,
        ds: Dataset,
        window: int = 1_000_000, 
        threshold: float = 0.2,
        step: Optional[int] = None, 
        groups='contig', 
        positions='pos', 
        scores=None,
        return_intervals=False, 
        target_chunk_size=None,
        **kwargs
    ):
        """Dask backend for LD matrix 
        
        See `api.ld_matrix` for documentation.

        Extra parameters added by this backend:

        Parameters
        ----------
        target_chunk_size : [type], optional
            Determines the approximate size of each chunk passed to blockwise computations.  
            This only controls the block heights while the widths are fixed by the number of 
            samples, so this height should be set accordingly for the underlying backend
            (e.g. cuda/numpy).
        """
        # This will coerce any dataset provided to the correct format
        # (if the coercion is supported by whatever dataset type was given)
        ds = ds.to.genotype_count_dataset()

        # 2D array of allele counts
        x = ds.data

        # Resolve arguments to individual (optional) vectors
        def resolve(f):
            if f is None:
                return None
            if isinstance(f, str):
                return ds[f]
            return f
        groups = resolve(groups)
        positions = resolve(positions)
        scores = resolve(scores)

        if positions is not None and step is not None:
            raise ValueError(
                'One of `positions` or `step` should be set but not both '
                '(`step` is only applicable to fixed windows, not those based on physical positions).'
            )

        # Compute row intervals as well as any necessary overlap
        ais, cis = axis_intervals(
            n=x.shape[0], window=window, step=step, 
            groups=groups, positions=positions, 
            target_chunk_size=target_chunk_size
        ) 

        # Get partitions representing the result of each chunked computation, which is at least
        # broken up by `groups` if provided (possibly more if `target_chunk_size` set)
        partitions = get_ld_matrix_partitions(x, ais, cis, threshold, scores=scores, **kwargs)

        # Concatenate partitions noting that each partition represents ALL data
        # for any one group (e.g. chromosome)
        ldm = get_ld_matrix_dataframe(partitions)

        if return_intervals:
            return ldm, (ais, cis)
        else:
            return ldm