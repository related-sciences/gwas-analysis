"""I/O API"""
from ..dispatch import register_function
from ..core import isdstype, GenotypeCountDataset
from xarray import Dataset


PLINK_DOMAIN = 'io.plink'


@register_function(PLINK_DOMAIN)
def read_plink(path, backend=None, **kwargs):
    """Import PLINK dataset"""
    pass


def write_zarr(ds: Dataset, path, **kwargs):

    # TODO: determine where this type of automatic optimization should be done
    # Assign filters/compression based on dataset type
    if isdstype(ds, GenotypeCountDataset):
        if 'encoding' not in kwargs:
            from .codecs import PackGeneticBits
            kwargs['encoding'] = {'data': {'filters': [PackGeneticBits()]}}

    return ds.to_zarr(path, **kwargs)

