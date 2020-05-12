import xarray as xr
from xarray import Dataset, DataArray
from typing import Optional, Type, Union, Tuple
from ...core import VARIABLES
from ...dispatch import DaskBackend, dispatches_from
from .. import core
from . import numba_backend
import dask


@dispatches_from(core.axis_intervals)
def axis_intervals(
    ds: Dataset,
    **kwargs
) -> Tuple[DataArray, DataArray]:
    """Dask backend for interval calculations

    Note that `ds` is currently required.
    """
    if ds is None:
        raise ValueError('Dataset (`ds`) must be provided for chunked backend')

    data_vars = [VARIABLES.contig, VARIABLES.pos, VARIABLES.data]

    futures = [
        dask.delayed(numba_backend.axis_intervals)(gds, **kwargs)
        # TODO: This groupby is really slow -- it should probably be 
        # a rechunking and a map_blocks instead
        for g, gds in list(ds[data_vars].groupby(VARIABLES.contig))
    ]
    intervals = dask.compute(*futures)

    ais = xr.concat([r[0] for r in intervals], dim='axis')
    cis = xr.concat([r[1] for r in intervals], dim='chunk')
    return ais, cis
    

core.register_backend_function(DaskBackend)(axis_intervals)