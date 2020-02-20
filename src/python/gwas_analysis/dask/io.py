import numpy as np

# Create a wrapper around pysnptools Bed reader class so that dask can
# access slices directly from PLINK files
class BedArray(object):

    def __init__(self, bed):
        self.bed = bed
        self.shape = (bed.sid_count, bed.iid_count)
        self.dtype = np.uint8
        self.ndim = 2

    def __getitem__(self, idx):
        assert isinstance(idx, tuple)
        chunk = self.bed.__getitem__(idx[::-1]).read(dtype=np.float32)
        arr = chunk.val.T
        # Add one to leave allele count in [0, 3] (0 = missing)
        arr = np.nan_to_num(arr, nan=-1) + 1
        arr = arr.astype(np.uint8)
        return arr