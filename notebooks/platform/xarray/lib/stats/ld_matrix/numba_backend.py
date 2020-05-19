import numba
import numpy as np
import pandas as pd
import functools
from xarray import Dataset
from typing import Optional, Tuple
from ...dispatch import NumbaBackend, dispatches_from
from ...typing import DataFrame, BlockArray
from ..axis_intervals import (
    AIF_IDX_INDEX,
    AIF_IDX_START,
    AIF_IDX_STOP,
    AIF_IDX_COUNT
)
from .. import core
from .. import numba_math as nmath

@numba.njit(nogil=True)
def __process_block_preallocated(
    x, 
    ais, 
    cts, 
    scores, 
    offset, 
    out_idx, 
    out_res, 
    out_cmp
):
    # Result array indexer
    ri = 0 

    # "Task index" loop
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


def _process_block_preallocated(
    x, 
    ais, 
    ci, 
    offset,
    scores, 
    index_dtype, 
    value_dtype, 
    value_init=np.nan, 
    **kwargs
):
    """LD Matrix implementation with pre-allocated result arrays"""
    n_pairs = ci.count
    cts = np.cumsum(ais[:, AIF_IDX_COUNT])

    out_idx = np.empty((n_pairs, 2), dtype=index_dtype)
    out_res = np.full(n_pairs, value_init, dtype=value_dtype)

    if scores is not None:
        scores = np.asarray(scores)
        out_cmp = np.empty(n_pairs, dtype=np.int8)
    else:
        scores = np.empty(0)
        out_cmp = np.empty(0)

    ri = __process_block_preallocated(x, ais, cts, scores, offset, out_idx, out_res, out_cmp)
    assert ri == n_pairs

    return out_idx, out_res, out_cmp


@numba.njit(nogil=True)
def __process_block_unallocated(
    x, 
    ais, 
    scores, 
    offset, 
    index_dtype, 
    value_dtype, 
    threshold=np.nan
):
    rows = list()
    no_threshold = np.isnan(threshold)

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

            cmp = np.int8(0)
            if scores.shape[0] > 0:
                if scores[i1] > scores[i2]:
                    cmp = np.int8(1)
                elif scores[i1] < scores[i2]:
                    cmp = np.int8(-1)

            if no_threshold or res >= threshold:
                rows.append((
                    index_dtype(index), 
                    index_dtype(other), 
                    value_dtype(res), 
                    cmp
                ))
    return rows


def _process_block_unallocated(
    x,
    intervals,
    threshold,
    return_value, 
    index_dtype, 
    value_dtype, 
    value_init=np.nan,
    scores=None,
    **kwargs
):
    """LD Matrix implementation with on-demand result array allocation"""
    x, ais, _, offset = core._ld_matrix_block_args(x, intervals, scores)

    if scores is not None:
        scores = np.asarray(scores)
    else:
        scores = np.empty(0)

    threshold = threshold or np.nan
    index_dtype = np.dtype(index_dtype).type
    value_dtype = np.dtype(value_dtype).type
    res = __process_block_unallocated(x, ais, scores, offset, index_dtype, value_dtype, threshold=threshold)
    cols = [
        ('i', index_dtype), 
        ('j', index_dtype), 
        ('value', value_dtype), 
        ('cmp', np.int8)
    ]
    res = pd.DataFrame(res, columns=[c[0] for c in cols])
    for k, v in dict(cols).items():
        res[k] = res[k].astype(v)
    if scores.shape[0] == 0:
        res = res.drop('cmp', axis=1)
    if not return_value:
        res = res.drop('value', axis=1)
    return res

@dispatches_from(core._ld_matrix_block)
def _ld_matrix_block(
    *args,
    threshold: Optional[float]=None,
    index_dtype=np.int32, 
    value_dtype=np.float32, 
    preallocate: Optional[bool] = None,
    **kwargs
) -> DataFrame:
    """Numba backend for LD Matrix block computation.

    See `core._ld_matrix_block` for primary documentation.

    Parameters
    ----------
    index_dtype : np.dtype, optional
        Array index data type, by default np.int32
    value_dtype : np.dtype, optional
        R2 (correlation) value data type, by default np.float32
    preallocate : bool, optional
        Whether or not resulting LD comparisons are preallocated
        (True) or allocated dynamically (False).  If not provided,
        then this will be True if `threshold` was not given or if
        the value for it is <= .02.  This is faster when few comparisons
        are likely to be omitted from results but requires 
        substantially more memory.  This should be False when using
        any kind of meaningful R2 threshold.
    """
    if preallocate is None:
        preallocate = threshold is None or threshold <= .02
    if preallocate:
        fn = functools.partial(core._ld_matrix_block_impl, _process_block_preallocated)
    else:
        fn = _process_block_unallocated
    return fn(
        *args,
        threshold=threshold,
        index_dtype=index_dtype,
        value_dtype=value_dtype,
        **kwargs
    )


core.register_backend_function(NumbaBackend)(_ld_matrix_block)