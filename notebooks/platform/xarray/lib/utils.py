import xarray as xr
import numpy as np
from xarray.core.pycompat import dask_array_type
from typing import Any, Sequence
import inspect

def is_array(data: Any) -> bool:
    # Check for duck array type (cf. https://numpy.org/neps/nep-0018-array-function-protocol.html)
    # See: 
    # - https://github.com/pydata/xarray/blob/1416d5ae475c0875e7a5d76fa4a8278838958162/xarray/core/duck_array_ops.py#L152
    # - https://github.com/dask/dask/blob/241027d1e270c2793244d9f16e53903c5ea5bd20/dask/array/core.py#L3693 (asarray)
    # - https://github.com/dask/dask/blob/master/dask/array/core.py#L2757 (from_array)
    return isinstance(data, xr.DataArray) \
        or isinstance(data, dask_array_type) \
        or hasattr(data, "__array_function__") # Dask should eventually support this
    
def is_signature_compatible(fn, *args, **kwargs) -> bool:
    sig = inspect.signature(fn)
    try:
        sig.bind(*args, **kwargs)
        return True
    except TypeError:
        return False
    
# TODO: Is there a type hint that will cover duck arrays (rather than Any)?
def check_array(cls: Any, data: Any, types: Sequence[Any] = None, dims: Sequence[str] = None) -> Any:
    if types:
        if not any([np.issubdtype(data.dtype, typ) for typ in types]):
            types = [typ.__name__ for typ in types]
            raise TypeError(f'{cls} expects data type to be one of [{types}] (not type: {data.dtype})')
    if dims:
        if data.ndim != len(dims):
            raise ShapeError(f'{cls} expects {ndim}-dimensional data (not shape: {data.shape})')
    return data

def check_domain(cls: Any, data: Any, min_value: float, max_value: float) -> Any:
    rng = [data.min(), data.max()]
    if rng[0] < min_value or rng[1] > max_value:
        raise ValueError(f'{cls} expects values in range [{min_value}, {max_value}]) (not range: {rng})')
    return data

def try_cast_unsigned(data: xr.DataArray, dtype: Any = np.uint8) -> xr.DataArray:
    if np.issubdtype(data.dtype, np.signedinteger):
        if int(data.min()) < np.iinfo(dtype).min:
            raise ValueError(f'Integer array must be all >= {np.iinfo(dtype).min} for unsigned cast')
        if int(data.min()) > np.iinfo(dtype).max:
            raise ValueError(f'Integer array must be all <= {np.iinfo(dtype).max} for unsigned cast')
        return data.astype(dtype)
    return data