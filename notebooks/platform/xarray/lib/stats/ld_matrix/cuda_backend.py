from numba import cuda
import numpy as np
import xarray as xr
import contextlib
import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from typing import Tuple, Optional
from dask.distributed import Lock
from ...typing import BlockArray
from ...dispatch import CudaBackend, dispatches_from
from ..axis_intervals import (
    ChunkInterval,
    AIF_IDX_INDEX,
    AIF_IDX_START,
    AIF_IDX_STOP,
    AIF_IDX_COUNT
)
from .. import cuda_math as gmath
from .. import core

# Axis interval fields (`aif`) cannot be used directly in kernel,
# TODO: remove if imported constants don't cause an error again
# AIF_IDX_INDEX = aif.index('index')
# AIF_IDX_START = aif.index('start')
# AIF_IDX_STOP = aif.index('stop')
# AIF_IDX_COUNT = aif.index('count')

@cuda.jit
def __process_block(x, ais, cts, scores, offset, out_idx, out_res, out_cmp):
    # pylint: disable=not-callable
    ti = cuda.grid(1)

    if ti >= ais.shape[0]:
        return
        
    # Get offset into output array from which all results here are written
    # (this is the sum of count of comparisons for all intervals up to this point)
    oi = 0 if ti == 0 else cts[ti - 1]

    # Alternatively, this could come from ti + chunk_interval['min_index'] but
    # reference explicitly instead for greater readability
    index = ais[ti, AIF_IDX_INDEX]
    start = ais[ti, AIF_IDX_START]
    stop = ais[ti, AIF_IDX_STOP]

    for i, other in enumerate(range(start, stop)):
        # All indexes are specified for global array so they must be offset by the
        # absolute index of the first row provided in x (i.e. `min_start` for the chunk)
        i1, i2 = index - offset, other - offset
        assert 0 <= i1 < x.shape[0]
        assert 0 <= i2 < x.shape[0]

        if i1 == i2:
            res = 1.0
        else:
            res = gmath.r2(x[i1], x[i2])

        idx = oi + i    
        out_idx[idx, 0] = index
        out_idx[idx, 1] = other
        out_res[idx] = res
        if out_cmp.shape[0] > 0:
            if scores[i1] > scores[i2]:
                out_cmp[idx] = 1
            elif scores[i1] < scores[i2]:
                out_cmp[idx] = -1
            else:
                out_cmp[idx] = 0


def _process_block(x, ais, ci, scores, index_dtype, value_dtype, value_init=np.nan):
    ais = np.asarray(ais)
    ci = ChunkInterval(*np.asarray(ci))
    n_tasks = len(ais)
    n_pairs = ci.count
    offset = ci.min_start
    # Cumulative sum of comparisons in each interval offsets into result array
    cts = np.cumsum(ais[:, AIF_IDX_COUNT])

    x = cuda.to_device(x)
    cts = cuda.to_device(cts)
    ais = cuda.to_device(ais)

    if scores is not None:
        scores = cuda.to_device(scores)
        out_cmp = cuda.to_device(np.empty(n_pairs, dtype=np.int8))
    else:
        scores = cuda.to_device(np.empty(0))
        out_cmp = cuda.to_device(np.empty(0))

    out_idx = cuda.to_device(np.empty((n_pairs, 2), dtype=index_dtype))
    out_res = cuda.to_device(np.full(n_pairs, value_init, dtype=value_dtype))

    __process_block.forall(n_tasks)(x, ais, cts, scores, offset, out_idx, out_res, out_cmp)

    return out_idx.copy_to_host(), out_res.copy_to_host(), out_cmp.copy_to_host()

@dispatches_from(core._ld_matrix_block)
def _ld_matrix_block(
    *args,
    index_dtype=np.int32, 
    value_dtype=np.float32, 
    **kwargs
):
    """CUDA backend for LD Matrix block computation.

    See `core._ld_matrix_block` for primary documentation.

    Parameters
    ----------
    index_dtype : np.dtype, optional
        Array index data type, by default np.int32
    value_dtype : np.dtype, optional
        R2 (correlation) value data type, by default np.float32
    """
    # TODO: Find a better way to synchronize for external resources
    # This intermittently breaks with errors like "Failed to acquire" on lock.release():
    # ctx = Lock('gpu') if 'lock' in kwargs and kwargs['lock'] else contextlib.suppress(); with ctx: ...
    return core._ld_matrix_block_impl(
        _process_block,
        *args, 
        index_dtype=index_dtype,
        value_dtype=value_dtype,
        **kwargs
    )
    
core.register_backend_function(CudaBackend)(_ld_matrix_block)
