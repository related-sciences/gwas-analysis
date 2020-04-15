import xarray as xr
import numpy as np
import re
from xarray.core.pycompat import dask_array_type
from typing import Any, Sequence
import collections.abc


class ShapeError(Exception):
    pass


def is_array(data: Any) -> bool:
    # Check for duck array type (cf. https://numpy.org/neps/nep-0018-array-function-protocol.html)
    # See: 
    # - https://github.com/pydata/xarray/blob/1416d5ae475c0875e7a5d76fa4a8278838958162/xarray/core/duck_array_ops.py#L152
    # - https://github.com/dask/dask/blob/241027d1e270c2793244d9f16e53903c5ea5bd20/dask/array/core.py#L3693 (asarray)
    # - https://github.com/dask/dask/blob/master/dask/array/core.py#L2757 (from_array)
    return isinstance(data, xr.DataArray) \
        or isinstance(data, dask_array_type) \
        or hasattr(data, "__array_function__") # Dask should eventually support this


def to_snake_case(value):
    return re.sub(r'(?<!^)(?=[A-Z])', '_', value).lower()


def is_shape_match(arr, dims) -> bool:
    for dim, size in dims.items():
        if arr.sizes[dim] != size:
            return False
    return True


def check_array(cls: Any, data: Any, types: Sequence[Any] = None, dims: Sequence[str] = None) -> Any:
    if types:
        if isinstance(types, str):
            types = [np.dtype(types)]
        elif not isinstance(types, collections.abc.Sequence):
            types = [types]
        if not any([np.issubdtype(data.dtype, typ) for typ in types]):
            types = [typ.__name__ for typ in types]
            raise TypeError(f'{cls} expects data type to be one of [{types}] (not type: {data.dtype})')
    if dims:
        if data.ndim != len(dims):
            raise ShapeError(f'{cls} expects {len(dims)}-dimensional data (not shape: {data.shape})')
    return data


def check_domain(data: Any, min_value: float, max_value: float) -> Any:
    rng = [data.min(), data.max()]
    if rng[0] < min_value or rng[1] > max_value:
        raise ValueError(f'Values must be in range [{min_value}, {max_value}]) (min/max found: {rng})')
    return data


def get_base_module(obj: Any) -> str:
    # Identify base module using method in xarray core
    # See: 
    # - https://github.com/pydata/xarray/blob/df614b96082b38966a329b115082cd8dddf9fb29/xarray/core/common.py#L203
    # - https://github.com/dask/dask/blob/241027d1e270c2793244d9f16e53903c5ea5bd20/dask/array/core.py#L1297
    return type(obj).__module__.split('.')[0]
