import numba
import numpy as np
from xarray import Dataset
from typing import Optional, Tuple
from ...dispatch import NumbaBackend, dispatches_from
from ..axis_intervals import (
    ChunkInterval,
    AIF_IDX_INDEX,
    AIF_IDX_START,
    AIF_IDX_STOP,
    AIF_IDX_COUNT
)
from .. import core
from .. import numba_math as nmath

@numba.njit(nogil=True)
def __process_block(x, ais, cts, scores, offset, out_idx, out_res, out_cmp):
    # Result array indexer
    ri = 0 

    # "Task index" loop, cf. cuda kernel 
    for ti in range(ais.shape[0]):
        # Fetch index of target for interval as well as start
        # and stop indexes to compare to for that target
        index = ais[ti, AIF_IDX_INDEX]
        start = ais[ti, AIF_IDX_START]
        stop = ais[ti, AIF_IDX_STOP]

        for other in range(start, stop):
            # All indexes are specified for global array so they must be offset by the
            # absolute index of the first row provided in x (i.e. `min_start` for the chunk)
            i1, i2 = index - offset, other - offset
            assert 0 <= i1 < x.shape[0]
            assert 0 <= i2 < x.shape[0]

            if i1 == i2:
                res = 1.0
            else:
                res = nmath.r2(x[i1], x[i2])

            out_idx[ri, 0] = index
            out_idx[ri, 1] = other
            out_res[ri] = res
            
            if out_cmp.shape[0] > 0:
                if scores[i1] > scores[i2]:
                    out_cmp[ri] = 1
                elif scores[i1] < scores[i2]:
                    out_cmp[ri] = -1
                else:
                    out_cmp[ri] = 0
            ri += 1
    return ri


def _process_block(x, ais, ci, scores, index_dtype, value_dtype, value_init=np.nan):
    x = np.asarray(x)
    ais = np.asarray(ais)
    ci = ChunkInterval(*np.asarray(ci))
    n_pairs = ci.count
    offset = ci.min_start
    cts = np.cumsum(ais[:, AIF_IDX_COUNT])

    out_idx = np.empty((n_pairs, 2), dtype=index_dtype)
    out_res = np.full(n_pairs, value_init, dtype=value_dtype)

    if scores is not None:
        scores = np.asarray(scores)
        out_cmp = np.empty(n_pairs, dtype=np.int8)
    else:
        scores = np.empty(0)
        out_cmp = np.empty(0)

    ri = __process_block(x, ais, cts, scores, offset, out_idx, out_res, out_cmp)
    assert ri == n_pairs

    return out_idx, out_res, out_cmp

@dispatches_from(core._ld_matrix_block)
def _ld_matrix_block(
    *args,
    index_dtype=np.int32, 
    value_dtype=np.float32, 
    **kwargs
):
    """Numba backend for LD Matrix block computation.

    See `core._ld_matrix_block` for primary documentation.

    Parameters
    ----------
    index_dtype : np.dtype, optional
        Array index data type, by default np.int32
    value_dtype : np.dtype, optional
        R2 (correlation) value data type, by default np.float32
    """
    return core._ld_matrix_block_impl(
        _process_block,
        *args, 
        index_dtype=index_dtype,
        value_dtype=value_dtype,
        **kwargs
    )


core.register_backend_function(NumbaBackend)(_ld_matrix_block)