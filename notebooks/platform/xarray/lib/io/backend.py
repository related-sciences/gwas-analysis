"""I/O backend implementations and configuration"""
import functools
import abc
from lib import config
from lib import core
from lib import compat


# ----------------------------------------------------------------------
# Backend Implementations

class Backend(abc.ABC):

    @property
    @abc.abstractmethod
    def id(self):
        pass


class PySnpToolsBackend(Backend):

    id = 'pysnptools'

    def __init__(self, version='1.9'):
        if version not in ['1.9']:
            raise NotImplementedError('Only PLINK version 1.9 currently supported')
        self.version = version

    def read_fam(self, path, sep='\t'):
        import pandas as pd
        names = ['sample_id', 'fam_id', 'pat_id', 'mat_id', 'is_female', 'phenotype']
        return pd.read_csv(path + '.fam', sep=sep, names=names)

    def read_bim(self, path, sep=' '):
        import pandas as pd
        names = ['contig', 'variant_id', 'cm_pos', 'pos', 'allele_1', 'allele_2']
        return pd.read_csv(path + '.bim', sep=sep, names=names)

    def read_plink(self, path, chunks='auto', fam_sep='\t', bim_sep=' '):
        from .pysnptools import Bed, BedArray
        import dask.array as da
        import xarray as xr

        # Load genotyping data
        # * Make sure to use asarray=False in order for masked arrays to propagate
        arr = da.from_array(
            BedArray(Bed(path, count_A1=True), dtype='int8'),
            chunks=chunks, lock=False, asarray=False, name=None
        )
        ds = core.create_genotype_count_dataset(arr)

        # Attach variant/sample data
        ds_fam = xr.Dataset.from_dataframe(self.read_fam(path, sep=fam_sep).rename_axis('sample', axis='index'))
        ds_bim = xr.Dataset.from_dataframe(self.read_bim(path, sep=bim_sep).rename_axis('variant', axis='index'))

        # Merge
        return ds.merge(ds_fam).merge(ds_bim)


# ----------------------------------------------------------------------
# Backend Management

config.register_option('io.plink.backend', 'auto', doc="""
The default PLINK reader/writer backend. Available options: 'pysnptools', default is 'auto'
""")


BACKENDS = {cls.id: cls() for cls in Backend.__subclasses__()}

REQUIREMENTS = {
    # List of preferred backends by key
    'io.plink.backend': [
        # Backend ID -> Sequence of package requirements
        ('pysnptools', ['pysnptools'])
    ]
}


def register_backend(backend: Backend) -> None:
    BACKENDS[backend.id] = backend


def _all_compatible(packages):
    for pkg in packages:
        status = compat.check_package(pkg)
        if not status.installed or not status.compatible:
            return False
    return True


def get_backend(key: str, backend: str) -> Backend:
    if backend == 'auto':
        for be, pkgs in REQUIREMENTS[key]:
            if _all_compatible(pkgs):
                backend = be
                break
        if backend == 'auto':
            # If no requirements are met for any backend, prompt for install of
            # the highest priority backend (i.e. the first one)
            be, pkgs = REQUIREMENTS[key][0]
            raise ImportError(f'I/O backend "{be}" requires the following packages to be installed: {pkgs}')
    return BACKENDS[backend]


def dispatch_backend(key):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            # Determine backend from id and eliminate from arguments
            backend = get_backend(key, kwargs.pop('backend', config.get_option(key)))
            return getattr(backend, fn.__name__)(*args, **kwargs)
        return wrapper
    return decorator

# ----------------------------------------------------------------------
# IO API


@dispatch_backend('io.plink.backend')
def read_plink(path, backend=None, **kwargs):
    """Import PLINK dataset"""
    pass
