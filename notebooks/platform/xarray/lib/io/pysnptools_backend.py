import numpy as np
from numcodecs import PackBits
from numcodecs.abc import Codec
from numcodecs.compat import ensure_ndarray, ndarray_copy
from ..core import create_genotype_count_dataset
from .core import register_backend, PLINKBackend
from ..compat import Requirement


class BedArray(object):

    def __init__(self, bed, dtype=np.uint8):
        self.bed = bed
        self.shape = (bed.sid_count, bed.iid_count)
        self.dtype = dtype
        self.ndim = 2

    def __getitem__(self, idx):
        assert isinstance(idx, tuple)
        # TODO: Make sure this is getting closed
        # https://github.com/fastlmm/PySnpTools/blob/0528bf8684a3dbd2eafc26d3fda34f6f902fce6c/pysnptools/snpreader/bed.py#L91
        chunk = self.bed.__getitem__(idx[::-1]).read(dtype=np.float32)
        arr = chunk.val.T
        arr = np.ma.masked_invalid(arr)
        arr = arr.astype(self.dtype)
        return arr


class PackGeneticBits(PackBits):
    """Custom Zarr plugin for encoding allele counts as 2 bit integers"""

    codec_id = 'packgeneticbits'

    def __init__(self):
        super().__init__()

    def encode(self, buf):

        # normalise input
        arr = ensure_ndarray(buf)

        # broadcast to 3rd dimension having 2 elements, least significant
        # bit then second least, in that order; results in nxmx2 array
        # containing individual bits as bools
        # see: https://stackoverflow.com/questions/22227595/convert-integer-to-binary-array-with-suitable-padding
        arr = arr[: ,: ,None] & (1 << np.arange(2)) > 0

        return super().encode(arr)

    def decode(self, buf, out=None):

        # normalise input
        enc = ensure_ndarray(buf).view('u1')

        # flatten to simplify implementation
        enc = enc.reshape(-1, order='A')

        # find out how many bits were padded
        n_bits_padded = int(enc[0])

        # apply decoding
        dec = np.unpackbits(enc[1:])

        # remove padded bits
        if n_bits_padded:
            dec = dec[:-n_bits_padded]

        # view as boolean array
        dec = dec.view(bool)

        # given a flattened version of what was originally an nxmx2 array,
        # reshape to group adjacent bits in second dimension and
        # convert back to int based on each little-endian bit pair
        dec = np.packbits(dec.reshape((-1, 2)), bitorder='little', axis=1)
        dec = dec.squeeze(axis=1)

        # handle destination
        return ndarray_copy(dec, out)


# Keep this in mind -- not sure where this sort of thing should go yet:
# from dask.distributed import WorkerPlugin
# class DaskCodecPlugin(WorkerPlugin):
#
#     def setup(self, worker):
#         from numcodecs.registry import register_codec
#         register_codec(PackGeneticBits)


class PySnpToolsBackend(PLINKBackend):

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
        import dask.array as da
        import xarray as xr
        from pysnptools.snpreader import Bed

        # Load genotyping data
        # * Make sure to use asarray=False in order for masked arrays to propagate
        arr = da.from_array(
            BedArray(Bed(path, count_A1=True), dtype='int8'),
            chunks=chunks, lock=False, asarray=False, name=None
        )
        ds = create_genotype_count_dataset(arr)

        # Attach variant/sample data
        ds_fam = xr.Dataset.from_dataframe(self.read_fam(path, sep=fam_sep).rename_axis('sample', axis='index'))
        ds_bim = xr.Dataset.from_dataframe(self.read_bim(path, sep=bim_sep).rename_axis('variant', axis='index'))

        # Merge
        return ds.merge(ds_fam).merge(ds_bim)

    def requirements(self):
        return [Requirement('pysnptools')]


register_backend(PySnpToolsBackend())
