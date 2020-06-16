import dask.array as da
import numpy as np
from pybgen import PyBGEN
from pybgen.parallel import ParallelPyBGEN
import xarray as xr

from .. import core
from ..typing import PathType
from ..compat import Requirement
from ..dispatch import ClassBackend, register_backend
from .core import BGEN_DOMAIN


def _array_name(f, path):
    return f.__qualname__ + ':' + str(path)


class BgenReader(object):

    def __init__(self, path, dtype=np.float32):
        self.path = str(path) # pybgen needs string paths

        # Use ParallelPyBGEN only to get all the variant seek positions from the BGEN index.
        # No parallel IO happens here.
        with ParallelPyBGEN(self.path) as bgen:
            bgen._get_all_seeks()
            self._seeks = bgen._seeks
            n_variants = bgen.nb_variants
            n_samples = bgen.nb_samples

            self.shape = (n_variants, n_samples, 3)
            self.dtype = dtype
            self.ndim = 3

            self.sample_id = bgen.samples
            # This may need chunking for large numbers of variants
            variants = list(bgen.iter_variant_info())
            self.variant_id = [v.name for v in variants]
            self.contig = [v.chrom for v in variants]
            self.pos = [v.pos for v in variants]
            self.a1 = [v.a1 for v in variants]
            self.a2 = [v.a2 for v in variants]

    def __getitem__(self, idx):
        if not isinstance(idx, tuple):
            raise IndexError(f'Indexer must be tuple (received {type(idx)})')
        if len(idx) != self.ndim:
            raise IndexError(f'Indexer must be two-item tuple (received {len(idx)} slices)')

        # Restrict to seeks for this chunk
        seeks_for_chunk = self._seeks[idx[0]]
        if len(seeks_for_chunk) == 0:
            return np.empty((0, 0), dtype=self.dtype)
        with PyBGEN(self.path, probs_only=True) as bgen:
            p = [probs for (_, probs) in bgen._iter_seeks(seeks_for_chunk)]
            return np.stack(p)[:,idx[1]]


@register_backend(BGEN_DOMAIN)
class PyBgenBackend(ClassBackend):

    id = 'pybgen'

    def read_bgen(self, path: PathType, chunks='auto', lock=False):

        bgen_reader = BgenReader(path)

        vars = {
            "sample_id": xr.DataArray(np.array(bgen_reader.sample_id), dims=["sample"]),
            "variant_id": xr.DataArray(np.array(bgen_reader.variant_id), dims=["variant"]),
            "contig": xr.DataArray(np.array(bgen_reader.contig), dims=["variant"]),
            "pos": xr.DataArray(np.array(bgen_reader.pos), dims=["variant"]),
            "a1": xr.DataArray(np.array(bgen_reader.a1), dims=["variant"]),
            "a2": xr.DataArray(np.array(bgen_reader.a2), dims=["variant"]),
        }

        arr = da.from_array(
            bgen_reader,
            chunks=chunks,
            lock=lock,
            asarray=False,
            name=_array_name(self.read_bgen, path))

        # pylint: disable=no-member
        ds = core.create_genotype_probability_alt_dataset(arr)
        ds = ds.assign(vars)
        return ds

    @property
    def requirements(self):
        return [Requirement('pybgen'), Requirement('dask')]
