"""Disposable module with temporary work-arounds until upstream changes are made in Dask/Xarray"""
import warnings
import xarray as xr
import dask.array as da
from dask.dataframe import DataFrame
from dask.array import Array
from dask.array.overlap import overlap, trim_internal
from numbers import Integral
# pylint: disable=no-name-in-module
from tlz import merge, pipe, concat, partial, get


def map_overlap(
    func, *args, depth=None, boundary=None, trim=True, align_arrays=True, **kwargs
):
    """ Map a function over blocks of arrays with some overlap

    We share neighboring zones between blocks of the array, map a
    function, and then trim away the neighboring strips.

    Parameters
    ----------
    func: function
        The function to apply to each extended block
    args : dask arrays
    depth: int, tuple, dict or list
        The number of elements that each block should share with its neighbors
        If a tuple or dict then this can be different per axis.
        If a list then each element of that list must be an int, tuple or dict
        defining depth for the corresponding array in `args`.
        Asymmetric depths may be specified using a dict value of (-/+) tuples.
        Note that asymmetric depths are currently only supported when
        ``boundary`` is 'none'.
        The default value is 0.
    boundary: str, tuple, dict or list
        How to handle the boundaries.
        Values include 'reflect', 'periodic', 'nearest', 'none',
        or any constant value like 0 or np.nan.
        If a list then each element must be a str, tuple or dict defining the
        boundary for the corresponding array in `args`.
        The default value is 'reflect'.
    trim: bool
        Whether or not to trim ``depth`` elements from each block after
        calling the map function.
        Set this to False if your mapping function already does this for you
    align_arrays: bool
        Whether or not to align chunks along equally sized dimensions when
        multiple arrays are provided.  This allows for larger chunks in some
        arrays to be broken into smaller ones that match chunk sizes in other
        arrays such that they are compatible for block function mapping. If
        this is false, then an error will be thrown if arrays do not already
        have the same number of blocks in each dimensions.
    **kwargs:
        Other keyword arguments valid in ``map_blocks``

    Examples
    --------
    >>> import numpy as np
    >>> import dask.array as da

    >>> x = np.array([1, 1, 2, 3, 3, 3, 2, 1, 1])
    >>> x = da.from_array(x, chunks=5)
    >>> def derivative(x):
    ...     return x - np.roll(x, 1)

    >>> y = x.map_overlap(derivative, depth=1, boundary=0)
    >>> y.compute()
    array([ 1,  0,  1,  1,  0,  0, -1, -1,  0])

    >>> x = np.arange(16).reshape((4, 4))
    >>> d = da.from_array(x, chunks=(2, 2))
    >>> d.map_overlap(lambda x: x + x.size, depth=1).compute()
    array([[16, 17, 18, 19],
           [20, 21, 22, 23],
           [24, 25, 26, 27],
           [28, 29, 30, 31]])

    >>> func = lambda x: x + x.size
    >>> depth = {0: 1, 1: 1}
    >>> boundary = {0: 'reflect', 1: 'none'}
    >>> d.map_overlap(func, depth, boundary).compute()  # doctest: +NORMALIZE_WHITESPACE
    array([[12,  13,  14,  15],
           [16,  17,  18,  19],
           [20,  21,  22,  23],
           [24,  25,  26,  27]])
    """
    # Look for invocation using deprecated single-array signature
    # map_overlap(x, func, depth, boundary=None, trim=True, **kwargs)
    if isinstance(func, Array) and callable(args[0]):
        warnings.warn(
            "Detected use of signature map_overlap(x, func) rather than "
            "map_overlap(func, *args) for multi-array support. Arguments "
            "will be swapped in this case but such an exception will not "
            "be made in a future release.",
            FutureWarning,
        )
        sig = ["func", "depth", "boundary", "trim"]
        depth = get(sig.index("depth"), args, depth)
        boundary = get(sig.index("boundary"), args, boundary)
        trim = get(sig.index("trim"), args, trim)
        func, args = args[0], [func]

    if not callable(func):
        raise TypeError(
            "First argument must be callable function, not {}\n"
            "Usage:   da.map_overlap(function, x)\n"
            "   or:   da.map_overlap(function, x, y, z)".format(type(func).__name__)
        )
    if not all(isinstance(x, Array) for x in args):
        raise TypeError(
            "All variadic arguments must be arrays, not {}\n"
            "Usage:   da.map_overlap(function, x)\n"
            "   or:   da.map_overlap(function, x, y, z)".format(
                [type(x).__name__ for x in args]
            )
        )

    # Coerce depth and boundary arguments to lists of individual
    # specifications for each array argument
    def coerce(xs, arg, fn):
        if not isinstance(arg, list):
            arg = [arg] * len(xs)
        return [fn(x.ndim, a) for x, a in zip(xs, arg)]

    depth = coerce(args, depth, coerce_depth)
    boundary = coerce(args, boundary, coerce_boundary)

    # Align chunks in each array to a common size
    if align_arrays:
        # Reverse unification order to allow block broadcasting
        inds = [list(reversed(range(x.ndim))) for x in args]
        _, args = da.core.unify_chunks(*list(concat(zip(args, inds))), warn=False)

    for i, x in enumerate(args):
        for j in range(x.ndim):
            if isinstance(depth[i][j], tuple) and boundary[i][j] != "none":
                raise NotImplementedError(
                    "Asymmetric overlap is currently only implemented "
                    "for boundary='none', however boundary for dimension "
                    "{} in array argument {} is {}".format(j, i, boundary[i][j])
                )

    def assert_int_chunksize(xs):
        assert all(type(c) is int for x in xs for cc in x.chunks for c in cc)

    assert_int_chunksize(args)
    args = [overlap(x, depth=d, boundary=b) for x, d, b in zip(args, depth, boundary)]
    assert_int_chunksize(args)
    x = da.map_blocks(func, *args, **kwargs)
    assert_int_chunksize([x])
    if trim:
        # Find index of array argument with maximum rank and break ties by choosing first provided
        i = sorted(enumerate(args), key=lambda v: (v[1].ndim, -v[0]))[-1][0]
        # Trim using depth/boundary setting for array of highest rank
        return trim_internal(x, depth[i], boundary[i])
    else:
        return x


def coerce_depth(ndim, depth):
    default = 0
    if depth is None:
        depth = default
    if isinstance(depth, Integral):
        depth = (depth,) * ndim
    if isinstance(depth, tuple):
        depth = dict(zip(range(ndim), depth))
    if isinstance(depth, dict):
        for i in range(ndim):
            if i not in depth:
                depth.update({i: 0})
    return depth


def coerce_boundary(ndim, boundary):
    default = "reflect"
    if boundary is None:
        boundary = default
    if not isinstance(boundary, (tuple, dict)):
        boundary = (boundary,) * ndim
    if isinstance(boundary, tuple):
        boundary = dict(zip(range(ndim), boundary))
    if isinstance(boundary, dict):
        for i in range(ndim):
            if i not in boundary:
                boundary.update({i: default})
    return boundary


def dataframe_to_dataset(df: DataFrame, dim, compute_lengths=True, add_coord=False):
    # Watch: https://github.com/pydata/xarray/issues/3929
    data_vars = {}
    for c in df:
        # Convert Dask Series to xr.DataArray with no coordinates
        # Use lengths=True to include chunk size calculations
        arr = df[c].to_dask_array(lengths=True if compute_lengths else None)
        data_vars[c] = xr.DataArray(arr, dims=dim)
    coords = {dim: da.arange(len(df))} if add_coord else None
    return xr.Dataset(data_vars, coords=coords)