import dask.array as da
import numpy as np
from pathlib import Path
import xarray as xr

from bgen_reader._bgen_file import bgen_file
from bgen_reader._bgen_metafile import bgen_metafile
from bgen_reader._metafile import create_metafile
from bgen_reader._reader import _infer_metafile_filepath
from bgen_reader._samples import generate_samples, read_samples_file

from .. import core
from ..typing import PathType
from ..compat import Requirement
from ..dispatch import ClassBackend, register_backend
from .core import BGEN_DOMAIN


def _array_name(f, path):
    return f.__qualname__ + ':' + str(path)


class BgenReader(object):

    def __init__(self, path, dtype=np.float32):
        self.path = Path(path)

        self.metafile_filepath = _infer_metafile_filepath(Path(self.path))
        if not self.metafile_filepath.exists():
            create_metafile(path, self.metafile_filepath, verbose=False)

        with bgen_metafile(self.metafile_filepath) as mf:
            self.n_variants = mf.nvariants
            self.npartitions = mf.npartitions
            self.partition_size = mf.partition_size

            # This may need chunking for large numbers of variants
            variants_df = mf.create_variants().compute()
            self.variant_id = variants_df["id"].tolist()
            self.contig = variants_df["chrom"].tolist()
            self.pos = variants_df["pos"].tolist()
            allele_ids = variants_df["allele_ids"].tolist()
            self.a1, self.a2 = tuple(zip(*[id.split(",") for id in allele_ids]))

        with bgen_file(self.path) as bgen:
            sample_path = self.path.with_suffix('.sample')
            if sample_path.exists():
                self.samples = read_samples_file(sample_path, verbose=False)
            else:
                if bgen.contain_samples:
                    self.samples = bgen.read_samples()
                else:
                    self.samples = generate_samples(bgen.nsamples)

        self.shape = (self.n_variants, len(self.samples), 3)
        self.dtype = dtype
        self.ndim = 3

    def __getitem__(self, idx):
        if not isinstance(idx, tuple):
            raise IndexError(f'Indexer must be tuple (received {type(idx)})')
        if len(idx) != self.ndim:
            raise IndexError(f'Indexer must be two-item tuple (received {len(idx)} slices)')

        if idx[0].start == idx[0].stop:
            return np.empty((0, 0), dtype=self.dtype)

        start_partition = idx[0].start // self.partition_size
        start_partition_offset = idx[0].start % self.partition_size
        end_partition = (idx[0].stop - 1) // self.partition_size
        end_partition_offset = (idx[0].stop - 1) % self.partition_size

        all_vaddr = []
        with bgen_metafile(self.metafile_filepath) as mf:
            for i in range(start_partition, end_partition + 1):
                partition = mf.read_partition(i)
                start_offset = start_partition_offset if i == start_partition else 0
                end_offset = end_partition_offset + 1 if i == end_partition else self.partition_size
                vaddr = partition["vaddr"].tolist()
                all_vaddr.extend(vaddr[start_offset:end_offset])

        with bgen_file(self.path) as bgen:
            genotypes = [bgen.read_genotype(vaddr) for vaddr in all_vaddr]
            probs = [genotype["probs"] for genotype in genotypes]
            return np.stack(probs)[:,idx[1]]


@register_backend(BGEN_DOMAIN)
class BgenReaderBackend(ClassBackend):

    id = 'bgen-reader'

    def read_bgen(self, path: PathType, chunks='auto', lock=False):

        bgen_reader = BgenReader(path)

        vars = {
            "sample_id": xr.DataArray(np.array(bgen_reader.samples), dims=["sample"]),
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
        return [Requirement('bgen-reader'), Requirement('dask')]
