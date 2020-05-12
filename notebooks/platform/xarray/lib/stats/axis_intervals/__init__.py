import collections

# Note: dataclass not supported by numba: https://github.com/numba/numba/issues/4037
axis_interval_fields = ['group', 'index', 'start', 'stop', 'count']
AxisInterval = collections.namedtuple('AxisInterval', axis_interval_fields)

chunk_interval_fields = ['group', 'min_index', 'max_index', 'min_start', 'max_stop', 'count']
ChunkInterval = collections.namedtuple('ChunkInterval', chunk_interval_fields)


from . import numba_backend

# Make available when done/necessary
try:
    from . import dask_backend
except ImportError:
    pass