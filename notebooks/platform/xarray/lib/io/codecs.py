from typing import Optional

import numpy as np
from numcodecs import PackBits
from numcodecs.compat import ensure_ndarray, ndarray_copy
from numcodecs.registry import register_codec


class PackGeneticBits(PackBits):
    """
    Custom Zarr plugin for encoding allele counts as 2 bit integers.

    012 coding system must be used. 0 if the individual is homozygous for the more frequent allele,
    1 if it is heterozygous, or 2 if it is homozygous for the less frequent allele. Use -1 for
    missing data. Any other coding systems might lead to invalid encoding.
    """

    codec_id = "packgeneticbits"
    __bit_mask = np.array([1, 2])

    def encode(self, buf: np.ndarray) -> np.ndarray:
        """Encodes a 2 dimensional array (variant x sample) of allele counts"""
        # normalise input
        arr = ensure_ndarray(buf)

        # broadcast to 3rd dimension having 2 elements, least significant
        # bit then second least, in that order; results in nxmx2 array
        # containing individual bits as bools
        # see: https://stackoverflow.com/questions/22227595/convert-integer-to-binary-array-with-suitable-padding
        arr = arr[:, :, None] & self.__bit_mask
        bool_arr = arr > 0
        # NOTE: -1 codes as [True, True] which equals to 3 in uint8 bit encoding

        return super().encode(bool_arr)

    def decode(self, buf: np.ndarray, out: Optional[np.ndarray] = None) -> np.ndarray:
        dec = super().decode(buf, out=None)

        # given a flattened version of what was originally an nxmx2 array,
        # reshape to group adjacent bits in second dimension and
        # convert back to int based on each little-endian bit pair
        dec = np.packbits(dec.reshape((-1, 2)), bitorder="little", axis=1)
        dec = dec.squeeze(axis=1)

        # -1 which codes for missing data got encoded as 11000000 which codes for 3 in uint8,
        # we revert that here
        dec[dec == 3] = -1

        # handle destination
        return ndarray_copy(dec, out)


register_codec(PackGeneticBits)
