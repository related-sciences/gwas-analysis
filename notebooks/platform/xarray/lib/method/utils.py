import collections
from numba import njit, typeof
from numba.typed import List

# Note: dataclass not supported by numba: https://github.com/numba/numba/issues/4037
AxisInterval = collections.namedtuple('AxisInterval', ['index', 'start', 'stop', 'count'])
ChunkInterval = collections.namedtuple('ChunkInterval', ['min_index', 'max_index', 'min_start', 'max_stop', 'count'])


# ----------------------------------------------------------------------
# Interval Calculations


# Interval tuple factory methods
# * workaround for lack of Numba support for namedtuple.defaults and None


@njit
def _axis_interval():
    return AxisInterval(0, 0, 0, 0)


@njit
def _chunk_interval():
    return ChunkInterval(0, 0, 0, 0, 0)


@njit
def _rnglen(start_index, stop_index):
    """Index interval size"""
    assert stop_index >= start_index
    # Stop index is exclusive
    return stop_index - start_index 


@njit
def _index_interval_right(n, i, window, positions):
    """Create index-based interval for items in window to the right of target index"""
    j = min(i + window, n)
    # Count is number of rows in interval including endpoints
    return AxisInterval(i, i, j, _rnglen(i, j))


@njit
def _position_interval_right(n, i, window, positions):
    """Create positions-based interval for items in window to the right of target position"""
    j = i + 1
    while j < n:
        if abs(positions[j] - positions[i]) > window:
            break
        j += 1
    return AxisInterval(i, i, j, _rnglen(i, j))


@njit
def _update_chunk_interval(chunk_interval, axis_interval):
    ci, ai = chunk_interval, axis_interval
    undefined = ci.count == 0
    return ChunkInterval(
        min_index=ai.index if undefined or ai.index < ci.min_index else ci.min_index,
        max_index=ai.index if undefined or ai.index > ci.max_index else ci.max_index,
        min_start=ai.start if undefined or ai.start < ci.min_start else ci.min_start,
        max_stop=ai.stop if undefined or ai.stop > ci.max_stop else ci.max_stop,
        count=ci.count + ai.count
    )


@njit
def axis_intervals(n, window, positions=None, target_chunk_size=None):
    axis_intervals, chunk_intervals = List(), List()
    chunk_interval = _chunk_interval()

    for i in range(n):
        # TODO: Figure out if numba would inline these functions if set conditionally
        # to a variable rather than switched on in if statement
        axis_interval = _axis_interval()
        if positions is None or len(positions) == 0:
            axis_interval = _index_interval_right(n, i, window, positions)
        else:
            if i + 1 < n:
                # Positions must be pre-sorted for intervals to be computed correctly
                if positions[i] > positions[i+1]:
                    raise ValueError('Positions must increase monotonically')
            axis_interval = _position_interval_right(n, i, window, positions)
        axis_intervals.append(axis_interval)

        # Update running chunk definition
        chunk_interval = _update_chunk_interval(chunk_interval, axis_interval)
        if target_chunk_size is None:
            continue

        # If each chunk has a target size, begin creating a new chunk if the current 
        # chunk meets or exceeds that size
        if _rnglen(chunk_interval.min_start, chunk_interval.max_stop + 1) >= target_chunk_size:
            chunk_intervals.append(chunk_interval)
            chunk_interval = _chunk_interval()

    # Add the running chunk definition if not already added
    if chunk_interval.count is not None:
        chunk_intervals.append(chunk_interval)

    return axis_intervals, chunk_intervals
