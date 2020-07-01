import numpy as np
import xarray as xr
from pathlib import Path
from .. import core
from ..typing import PathType
from ..compat import Requirement
from ..dispatch import ClassBackend, register_backend
from ..dask_ext import dataframe_to_dataset
from .core import PLINK_DOMAIN

from sgkit import create_genotype_call_dataset


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
        self.shape = (n_sid, n_iid, 2)
        self.dtype = dtype
        self.ndim = 3

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
        # Add a ploidy dimension, so allele counts of 0, 1, 2 correspond to 00, 01, 11
        arr2 = np.empty((arr.shape[0], arr.shape[1], 2), dtype=self.dtype)
        arr2[:,:,0] = np.where(arr == 2, 1, 0) 
        arr2[:,:,1] = np.where(arr == 0, 0, 1) 
        return arr2

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

    def read_plink(self, path: PathType, chunks='auto', fam_sep='\t', bim_sep=' ', count_A1=True, lock=False):
        import dask.array as da

        # TODO: Have this choose matching chunk sizes for data array and data frames
        # (to avoid zarr write errors)

        # Load axis data first to determine dimension sizes
        df_fam = self.read_fam(path, sep=fam_sep)
        df_bim = self.read_bim(path, sep=bim_sep)

        # Load genotyping data
        call_gt = da.from_array(
            # Make sure to use asarray=False in order for masked arrays to propagate
            BedReader(path, (len(df_bim), len(df_fam)), count_A1=count_A1),
            chunks=chunks, 
            # Lock must be true with multiprocessing dask scheduler
            # to not get pysnptools errors (it works w/ threading backend though)
            lock=lock, 
            asarray=False,
            name=_array_name(self.read_plink, path)
        )

        # TODO: either avoid computing Dask arrays, or just use Pandas
        df_bin_pd = df_bim.compute()
        df_fam_pd = df_fam.compute()

        variant_contig_names = df_bin_pd["contig"].values
        # TODO: can we get the names from somewhere in a given order? (since following sorts them)
        u, variant_contig = np.unique(variant_contig_names, return_inverse=True)

        variant_pos = df_bin_pd["pos"].values

        a1 = df_bin_pd["a1"].values
        a2 = df_bin_pd["a2"].values
        variant_alleles = np.column_stack((a1, a2))
        variant_alleles = variant_alleles.astype(np.dtype("S1"))

        variant_id = df_bin_pd["variant_id"].values
        variant_id = variant_id.astype(str)

        sample_id = df_fam_pd["fam_id"].values # not sure it's labelled correctly, fam_id should be sample_id
        sample_id = sample_id.astype(str)

        return create_genotype_call_dataset(
            variant_contig,
            variant_pos,
            variant_alleles,
            sample_id,
            call_gt=call_gt,
            variant_id=variant_id
        )

    @property
    def requirements(self):
        return [Requirement('pysnptools'), Requirement('dask')]
