import numpy as np
from numcodecs import PackBits
from numcodecs.compat import ensure_ndarray, ndarray_copy
from numcodecs.registry import register_codec


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


register_codec(PackGeneticBits)