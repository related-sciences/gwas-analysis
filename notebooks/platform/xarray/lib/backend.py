import numpy as np
import xarray as xr
from xarray.core.pycompat import dask_array_type, sparse_array_type

def try_import(fn: callable) -> bool:
    try:
        fn()
        return True
    except ImportError:
        return False

numpy_array_type = (np.ndarray,)
xarray_array_type = (xr.DataArray,)
    
def import_cuda():
    import numba.cuda.cudadrv.devicearray.DeviceNDArray
cuda_installed = try_import(import_cuda)

if cuda_installed:
    import numba.cuda.cudadrv.devicearray as ca
    cuda_array_type = (ca.DeviceNDArray,)
else:
    cuda_array_type = ()
    