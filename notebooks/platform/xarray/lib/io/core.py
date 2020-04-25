"""I/O backend implementations and configuration"""
from ..dispatch import FrontendDispatcher, ClassBackend, Domain
from ..core import isdstype, GenotypeCountDataset
from xarray import Dataset

DOMAIN = Domain('io')
PLINK_DOMAIN = DOMAIN.append('plink')

# ----------------------------------------------------------------------
# IO API


class IOBackend(ClassBackend):
    domain = DOMAIN


class PLINKBackend(IOBackend):
    domain = PLINK_DOMAIN


dispatchers = dict()


def dispatch(domain):
    if domain not in dispatchers:
        dispatchers[domain] = FrontendDispatcher(DOMAIN.append(domain))

    def decorator(fn):
        return dispatchers[domain].add(fn)
    return decorator


@dispatch('plink')
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

# ----------------------------------------------------------------------
# IO Backend Registration


def register_backend(backend: IOBackend):
    dispatchers[backend.domain[-1]].register(backend)
