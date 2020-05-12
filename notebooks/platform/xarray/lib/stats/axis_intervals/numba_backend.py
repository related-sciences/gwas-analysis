"""Genomic interval calculations"""
import collections
from numba import njit, typeof
from numba.typed import List
from typing import Optional, Type, Union, Tuple
from typing_extensions import Literal
import numpy as np
import xarray as xr
from xarray import Dataset, DataArray
from ...dispatch import NumbaBackend, dispatches_from
from .. import core
from ...core import VARIABLES
from . import (
    axis_interval_fields,  
    AxisInterval, 
    chunk_interval_fields, 
    ChunkInterval
)

# Make constant for numba
n_axis_interval_fields = len(axis_interval_fields)
n_chunk_interval_fields = len(chunk_interval_fields)


@njit
def _axis_interval():
    # Workaround for lack of numba support for namedtuple.defaults and None
    return AxisInterval(0, 0, 0, 0, -1)


@njit
def _chunk_interval():
    # Workaround for lack of numba support for namedtuple.defaults and None
    return ChunkInterval(0, 0, 0, 0, 0, -1)


@njit
def _rnglen(start_index, stop_index):
    """Index interval size"""
    assert stop_index >= start_index
    assert start_index >= 0
    # Stop index is exclusive
    return stop_index - start_index 


@njit
def _is_none(array):
    # Note: this boolean condition must include "is not None" check on `positions`
    # to avoid numba error 'Invalid use of Function(<built-in function getitem>) 
    # with argument(s) of type(s): (none, int64)' (it cannot be made a variable 
    # outside of loop)
    # https://github.com/numba/numba/issues/3585
    return array is None or len(array) == 0


@njit
def _clip(i, j, groups):
    # j is exclusive endpoint
    if _is_none(groups):
        return j
    for x in range(i + 1, j):
        if not _is_none(groups) and groups[i] != groups[x]:
            return x
    return j

@njit
def _index_interval_right(n, i, window, step, groups, positions):
    """Create index-based interval for items in window to the right of target index"""
    # Notes:
    # - A step of 1 disables its effects on the calculation
    # - The +1 ensures that each exclusive end range at least includes self 
    j = i + window - (i % step) + 1
    j = _clip(i, min(j, n), groups)
    return AxisInterval(
        group=groups[i] if not _is_none(groups) else 0,
        index=i,
        start=i,
        stop=j,
        count=_rnglen(i, j)
    )


@njit
def _position_interval_right(n, i, window, step, groups, positions):
    """Create positions-based interval for items in window to the right of target position"""
    j = i + 1
    while j < n:
        if abs(positions[j] - positions[i]) > window:
            break
        if not _is_none(groups) and groups[i] != groups[j]:
            break
        j += 1
    return AxisInterval(
        group=groups[i] if not _is_none(groups) else 0,
        index=i,
        start=i,
        stop=j,
        count=_rnglen(i, j)
    )

@njit
def _is_undefined(chunk_interval):
    return chunk_interval.count < 0


@njit
def _update_chunk_interval(chunk_interval, axis_interval):
    ci, ai = chunk_interval, axis_interval
    undefined = _is_undefined(ci)
    assert axis_interval.count > 0, \
        'Axis interval cannot be empty'
    assert undefined or ci.group == ai.group, \
        'Group for chunk interval must equal group for axis interval'
    return ChunkInterval(
        min_index=ai.index if undefined or ai.index < ci.min_index else ci.min_index,
        # Add one so that max_index, like max_stop, is exclusive
        max_index=ai.index + 1 if undefined or ai.index + 1 > ci.max_index else ci.max_index,
        min_start=ai.start if undefined or ai.start < ci.min_start else ci.min_start,
        max_stop=ai.stop if undefined or ai.stop > ci.max_stop else ci.max_stop,
        # Ensure that counts are always aggregated as 64 bit (they may get very large)
        count=max(ci.count, 0) + np.int64(ai.count),
        group=ai.group
    )


@njit
def _update_axis_intervals(i, axis_intervals, axis_interval):
    for j in range(n_axis_interval_fields):
        axis_intervals[i, j] = axis_interval[j]

@njit
def _to_array(chunk_intervals):
    n = len(chunk_intervals)
    cis = np.empty((n, n_chunk_interval_fields), dtype=np.int64)
    for i in range(n):
        for j in range(n_chunk_interval_fields):
            cis[i, j] = chunk_intervals[i][j]
    return cis

@njit
def __axis_intervals(n, window, step=None, groups=None, positions=None, target_chunk_size=None, dtype=np.int32):
    # window of 0 means do only self comparison, window of 1 means do one to right
    axis_intervals = np.empty((n, n_axis_interval_fields), dtype=dtype)
    chunk_intervals = List()

    chunk_interval = _chunk_interval()
    for i in range(n):
        inxt = min(i + 1, n - 1)

        # Define interval for current iterate along axis using either fixed window
        # size or a provided coordinate vector (as `positions`)
        axis_interval = _axis_interval()
        if not _is_none(positions):
            # Positions must be pre-sorted for intervals to be computed correctly
            groups_equal = _is_none(groups) or groups[i] == groups[inxt]
            if positions[i] > positions[inxt] and groups_equal:
                    raise ValueError('Positions must increase monotonically')
            axis_interval = _position_interval_right(n, i, window, step, groups, positions)
        else:
            axis_interval = _index_interval_right(n, i, window, step, groups, positions)
        _update_axis_intervals(i, axis_intervals, axis_interval)

        # Update running chunk definition based on interval for iterate
        chunk_interval = _update_chunk_interval(chunk_interval, axis_interval)

        # Start a new chunk interval if this is the last iterate in a group or
        # the current chunk size is as large as the target size 
        last_in_group = not _is_none(groups) and groups[i] != groups[inxt]
        last_in_chunk = target_chunk_size is not None and \
            _rnglen(chunk_interval.min_start, chunk_interval.max_stop) >= target_chunk_size
        if last_in_group or last_in_chunk:
            chunk_intervals.append(chunk_interval)
            chunk_interval = _chunk_interval()

    # Add the running chunk definition if not already added
    if not _is_undefined(chunk_interval):
        chunk_intervals.append(chunk_interval)

    return axis_intervals, _to_array(chunk_intervals)


def _to_dataarray(intervals, typ, dim):
    return xr.DataArray(intervals, coords={'var': list(typ._fields)}, dims=[dim, 'var'])


def _axis_intervals(
    window: int, 
    groups=None, 
    positions=None, 
    step: Optional[int]=1, 
    n: Optional[int]=None, 
    target_chunk_size: Optional[int]=None, 
    dtype: Type=np.int32
):
    """Get axis intervals for overlapping computations
    
    See `stats.core.axis_intervals` for documentation.
    """
    if window < 0:
        raise ValueError(f'`window` must be >= 0 (not {window})')
    if step is not None and step < 1:
        raise ValueError(f'`step` must be >= 1 (not {step})')
    if step is not None and step > window:
        raise ValueError(f'`step` must be >= `window` (not step={step}, window={window})')
    if n is None:
        n = next((len(v) for v in [groups, positions] if v is not None), None)
        if n is None:
            raise ValueError('At least one of `n`, `groups` or `positions` must be defined')
    if groups is not None and len(groups) != n:
        raise ValueError(f'Size of `groups` vector ({len(groups)}) must equal `n` ({n})')
    if positions is not None and len(positions) != n:
        raise ValueError(f'Size of `positions` vector ({len(positions)}) must equal `n` ({n})')
    if target_chunk_size is not None and target_chunk_size < 1:
        raise ValueError(f'`target_chunk_size` must be >= 1 (not {target_chunk_size})')

    if positions is not None and step is not None:
        raise ValueError(
            'One of `positions` or `step` should be set but not both '
            '(`step` is only applicable to fixed windows, not those based on physical positions).'
        )

    # Default step if not overriden to allow 0 window
    step = step or 1

    # Using array values of None leads to the following error:
    # 'Invalid use of Function(<built-in function getitem>) with argument(s) of type(s): (none, int64)'
    # This can be avoided by putting both an is None and a len > 0 condition in every check for existence, 
    # but empty arrays will be used instead (as sentinel values) for brevity
    # See: https://github.com/numba/numba/issues/3585
    def asarray(x):
        return np.empty(0, dtype=dtype) if x is None else np.asarray(x)
    groups = asarray(groups)
    positions = asarray(positions)

    # Compute intervals and associated chunks (as numpy arrays)
    ais, cis = __axis_intervals(
        n, window, step=step, groups=groups, positions=positions, 
        target_chunk_size=target_chunk_size, 
        # This must not be a string
        dtype=np.dtype(dtype)
    )

    # Convert to xarray
    ais = _to_dataarray(ais, AxisInterval, dim='axis')
    cis = _to_dataarray(cis, ChunkInterval, dim='chunk')
    
    return ais, cis

@dispatches_from(core.axis_intervals)
def axis_intervals(
    ds: Dataset,
    window: Optional[int]=None, 
    step: Optional[int]=1, 
    unit: Literal['index', 'physical']='index',
    n: Optional[int]=None, 
    target_chunk_size: Optional[int]=None, 
    dtype: Union[str, Type]='int32'
) -> Tuple[DataArray, DataArray]:
    ds = ds.to.genotype_count_dataset()
    groups = ds.get(VARIABLES.contig)
    positions = ds.get(VARIABLES.pos) if unit == 'physical' else None
    n = ds.data.shape[0]

    if positions is not None:
        step = None

    if n is None:
        raise ValueError('`n` must be provided for interval calculations')

    # Default to full intervals
    if window is None:
        window = n

    return _axis_intervals(
        window=window, 
        step=step,
        n=n,
        groups=groups,
        positions=positions,
        target_chunk_size=target_chunk_size,
        dtype=dtype
    )

core.register_backend_function(NumbaBackend)(axis_intervals)