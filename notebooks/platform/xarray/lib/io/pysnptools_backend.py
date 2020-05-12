import numpy as np
import xarray as xr
from pathlib import Path
from .. import core
from ..typing import PathType
from ..compat import Requirement
from ..dispatch import ClassBackend, register_backend
from ..dask_ext import dataframe_to_dataset
from .core import PLINK_DOMAIN


class BedReader(object):

    def __init__(self, path, shape, dtype=np.int8, count_A1=True):
        from pysnptools.snpreader import Bed
        # n variants (sid = SNP id), n samples (iid = Individual id)
        n_sid, n_iid = shape
        # Initialize Bed with empty arrays for axis data, otherwise it will
        # load the bim/map/fam files entirely into memory (it does not do out-of-core for those)
        self.bed = Bed(
            str(path), count_A1=count_A1,
            # Array (n_sample, 2) w/ FID and IID
            iid=np.empty((n_iid, 2), dtype='str'),
            # SNP id array (n_variants)
            sid=np.empty((n_sid,), dtype='str'),
            # Contig and positions array (n_variants, 3)
            pos=np.empty((n_sid, 3), dtype='int')
        )
        self.shape = (n_sid, n_iid)
        self.dtype = dtype
        self.ndim = 2

    @staticmethod
    def _is_empty_slice(s):
        return s.start == s.stop

    def __getitem__(self, idx):
        if not isinstance(idx, tuple):
            raise IndexError(f'Indexer must be tuple (received {type(idx)})')
        if len(idx) != self.ndim:
            raise IndexError(f'Indexer must be two-item tuple (received {len(idx)} slices)')

        # This is called by dask with empty slices before trying to read any chunks, so it may need
        # to be handled separately if pysnptools is slow here
        # if all(map(BedReader._is_empty_slice, idx)):
        #     return np.empty((0, 0), dtype=self.dtype)

        arr = self.bed[idx[::-1]].read(dtype=np.float32, view_ok=False).val.T
        arr = np.ma.masked_invalid(arr)
        arr = arr.astype(self.dtype)
        return arr

    def close(self):
        # This is not actually crucial since a Bed instance with no
        # in-memory bim/map/fam data is essentially just a file pointer
        # but this will still be problematic if the an array is created
        # from the same PLINK dataset many times
        self.bed._close_bed()


def _array_name(f, path):
    return f.__qualname__ + ':' + str(path)


# TODO: Make dask usage optional
@register_backend(PLINK_DOMAIN)
class PySnpToolsBackend(ClassBackend):

    id = 'pysnptools'

    def __init__(self, version='1.9'):
        if version not in ['1.9']:
            raise NotImplementedError('Only PLINK version 1.9 currently supported')
        self.version = version

    def read_fam(self, path: PathType, sep='\t'):
        import dask.dataframe as dd
        names = ['sample_id', 'fam_id', 'pat_id', 'mat_id', 'is_female', 'phenotype']
        return dd.read_csv(str(path) + '.fam', sep=sep, names=names, storage_options=dict(auto_mkdir=False))

    def read_bim(self, path: PathType, sep=' '):
        import dask.dataframe as dd
        names = ['contig', 'variant_id', 'cm_pos', 'pos', 'a1', 'a2']
        return dd.read_csv(str(path) + '.bim', sep=sep, names=names, storage_options=dict(auto_mkdir=False))

    def read_plink(self, path: PathType, chunks='auto', fam_sep='\t', bim_sep=' ', count_A1=True):
        import dask.array as da

        # TODO: Have this choose matching chunk sizes for data array and data frames
        # (to avoid zarr write errors)

        # Load axis data first to determine dimension sizes
        df_fam = self.read_fam(path, sep=fam_sep)
        df_bim = self.read_bim(path, sep=bim_sep)

        # Load genotyping data
        arr = da.from_array(
            # Make sure to use asarray=False in order for masked arrays to propagate
            BedReader(path, (len(df_bim), len(df_fam)), count_A1=count_A1),
            chunks=chunks, lock=False, asarray=False,
            name=_array_name(self.read_plink, path)
        )
        # pylint: disable=no-member
        ds = core.create_genotype_count_dataset(arr)

        # Create variant/sample datasets from dataframes
        ds_fam = dataframe_to_dataset(df_fam, dim='sample', compute_lengths=True, add_coord=False)
        ds_bim = dataframe_to_dataset(df_bim, dim='variant', compute_lengths=True, add_coord=False)

        # Merge
        return ds.merge(ds_fam).merge(ds_bim)

    @property
    def requirements(self):
        return [Requirement('pysnptools'), Requirement('dask')]
