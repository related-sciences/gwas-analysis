from __future__ import annotations
import numpy as np
from functools import partial
from multipledispatch import dispatch
from .compat import numpy_array_type, dask_array_type, xarray_array_type
from .utils import get_base_module

namespace = dict()
dispatch = partial(dispatch, namespace=namespace)

@dispatch(numpy_array_type)
def get_mask_array(data):
    mask = np.ma.getmask(data)
    return None if isinstance(mask, np.bool_) and not mask else mask

@dispatch(dask_array_type)
def get_mask_array(data):
    import dask.array as da
    # For some reason np.ma.getmaskarray does not work on masked dask arrays
    meta_module = get_base_module(data._meta)
    meta_name = type(data._meta).__name__
    if meta_module == 'numpy' and meta_name == 'MaskedArray':
        return da.ma.getmaskarray(data)
    else:
        return None
    
@dispatch(xarray_array_type)
def get_mask_array(data):
    # Xarray does not support masking
    return None


def _check_fill_value(fill_value, dtype):
    if fill_value < 0 and not np.issubdtype(dtype, np.signedinteger):
        raise ValueError(f'Mask filling with negative value not supported '
                         f'for unsigned integer types (fill_value = {fill_value}, dtype = {dtype})')
        
@dispatch(numpy_array_type)
def get_filled_array(data, fill_value):
    _check_fill_value(fill_value, data.dtype)
    return np.ma.filled(data, fill_value=fill_value)

@dispatch(dask_array_type)
def get_filled_array(data, fill_value):
    import dask.array as da
    _check_fill_value(fill_value, data.dtype)
    return da.ma.filled(data, fill_value=fill_value)
