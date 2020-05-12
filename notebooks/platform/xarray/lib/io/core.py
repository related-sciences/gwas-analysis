"""I/O API"""
from ..dispatch import register_function, Domain
from ..core import isdstype, GenotypeCountDataset
from . import DOMAIN
from xarray import Dataset
from ..typing import PathType

DOMAIN = Domain(DOMAIN)
PLINK_DOMAIN = DOMAIN.append('plink')


@register_function(PLINK_DOMAIN, append=False)
def read_plink(path, backend=None, **kwargs) -> Dataset:
    """Import PLINK dataset"""
    pass


def write_zarr(ds: Dataset, path: PathType, rechunk: bool=False, **kwargs) -> "ZarrStore":
    """Write dataset to zarr

    Xarray can be used directly to manage exports, but this function will
    add custom filters and chunk unification where appropriate first.

    Parameters
    ----------
    ds : Dataset
        Dataset to write
    path : PathType
        Path to write to on disk
    rechunk : bool, optional
        Rechunk all arrays as dask with common chunks along each dimensions, 
        by default False
    kwargs
        Keyword arguments for xr.Dataset.to_zarr
    """

    if rechunk:
        # Rechunk to primary data array
        ds = ds.chunk(dict(zip(ds.data.dims, ds.data.chunks)))

    if isdstype(ds, GenotypeCountDataset):
        if 'encoding' not in kwargs:
            from .codecs import PackGeneticBits
            kwargs['encoding'] = {'data': {'filters': [PackGeneticBits()]}}

    return ds.to_zarr(path, **kwargs)

