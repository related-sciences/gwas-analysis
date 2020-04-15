from __future__ import annotations
import xarray as xr
import numpy as np
import inspect
import sys
from dataclasses import dataclass
from lib import DIM_VARIANT, DIM_SAMPLE, DIM_PLOIDY, DIM_ALLELE
from lib.utils import is_array, check_array, check_domain, is_shape_match, to_snake_case
from lib.ops import get_mask_array, get_filled_array
from typing import Mapping, Sequence, Any, Type, Hashable
from xarray import Dataset
import collections
       
# TODO:
# - Is there a type hint compatible with duck arrays?
#   - Xarray uses Any (e.g. https://github.com/pydata/xarray/blob/master/xarray/core/dataarray.py#L559)
#   - See https://github.com/numpy/numpy/issues/7370 (PEP 484)
# - What is the best way to implement code dependent on optional dependencies (i.e. avoid requiring global dask imports)?
# - Are we sure it's worth assuming no support for masking in Xarray?
#   - https://github.com/pydata/xarray/issues/1194

# Notes:
# - xr.DataArray(xr.DataArray()) will call .asarray on input argument pushing .data into memory as numpy
# - Xarray often invokes constructors on DataArray/Dataset like this: ```type(self)(*args, **kwargs)```
#   - This means that subclasses must be able to differentiate between constructor params
# - Accessing "owner" class when working with descriptors is possible with __set_name__
#   - See: https://www.python.org/dev/peps/pep-0487/
#   - See: https://stackoverflow.com/questions/2366713/can-a-decorator-of-an-instance-method-access-the-class
# - Copying docstrings in wrapped functions: https://docs.python.org/2/library/functools.html#functools.update_wrapper

MISSING = -1


# ----------------- #
# Utility Functions #
# ----------------- #

def _mask(ds, fill_value=MISSING, var='data'):
    """Apply mask to dataset data variable"""
    is_masked = ds.get('is_masked')
    if is_masked is None:
        return ds[var]
    return xr.where(is_masked, fill_value, ds[var])


def _transmute(fn, ds, *args, **kwargs):
    """Create new dataset type with (overridable) default attribute propagation behavior"""
    # Preserve attributes
    kwargs['attrs'] = {**ds.attrs, **kwargs.get('attrs', {})}
    # Default any optional parameters to current values (these rarely change across dataset types)
    for p in inspect.signature(fn).parameters.values():
        # Only override data variables in result if not provided and present within the current dataset
        if p.default is None and p.name not in kwargs:
            kwargs[p.name] = ds.get(p.name)
    # Call factory method with positional (required) arguments as is, and any dynamic arguments as keyword args
    return fn(*args, **kwargs)


def _extract_array(data, is_masked):
    if is_masked is None:
        is_masked = get_mask_array(data)
        if is_masked is not None:
            data = get_filled_array(data, fill_value=MISSING)
    return data, is_masked


def _create_array(data, dims=None, set_coords=True, **kwargs):
    if isinstance(data, xr.DataArray):
        return data
    kwargs = dict(kwargs)
    if 'dims' not in kwargs and dims:
        kwargs['dims'] = dims
    if 'coords' not in kwargs and dims and set_coords:
        kwargs['coords'] = {dims[i]: np.arange(data.shape[i]) for i in range(data.ndim)}
    return xr.DataArray(data, **kwargs)


def _create_dataset(cls, arrays, **kwargs):
    # Validate arrays against constraints (types, dimensions, domains, etc.)
    arrays = {
        k: check_array(cls, v, types=cls.params[k].dtypes, dims=cls.params[k].dims)
        for k, v in arrays.items()
        if v is not None
    }

    # Infer masks provided within underlying arrays, if supported by backend
    if 'is_masked' in cls.params and 'data' in cls.params:
        arrays['data'], arrays['is_masked'] = _extract_array(arrays['data'], arrays.get('is_masked'))

    # Convert all arrays to xarray
    arrays = {
        k: _create_array(v, dims=cls.params[k].dims, **kwargs)
        for k, v in arrays.items()
        if v is not None
    }

    # Add dataset type to attributes
    kwargs = {'attrs': {**(kwargs.get('attrs', {})), **{'type': cls}}}
    return xr.Dataset(data_vars=arrays, **kwargs)

# -------------- #
# Dataset Models #
# -------------- #


@dataclass
class ArrayParameter:
    dims: Sequence[str]
    dtypes: Sequence[Type]


def _params(dims, dtypes, core_vars=['data'], flag_vars=['is_masked', 'is_phased']) -> Mapping[Hashable, ArrayParameter]:
    return {
        **{v: ArrayParameter(dims, dtypes) for v in core_vars},
        **{v: ArrayParameter([DIM_VARIANT, DIM_SAMPLE], [np.bool_]) for v in flag_vars}
    }


class GeneticDataset:
    pass


def transmutation(target_dstype):
    def decorator(fn):
        if not hasattr(GeneticDataset, 'transmutations'):
            GeneticDataset.transmutations = collections.defaultdict(dict)
        GeneticDataset.transmutations[fn.__qualname__.split('.')[0]][target_dstype.__name__] = fn
        return fn
    return decorator


class GenotypeDosageDataset(GeneticDataset):

    params: Mapping[Hashable, ArrayParameter] = _params([DIM_VARIANT, DIM_SAMPLE], [np.floating])

    @classmethod
    def create(cls, data: Any, is_phased: Any = None, is_masked: Any = None, **kwargs) -> Dataset:
        arrays = dict(data=data, is_phased=is_phased, is_masked=is_masked)
        return _create_dataset(cls, arrays, **kwargs)


class GenotypeCountDataset(GeneticDataset):

    params: Mapping[Hashable, ArrayParameter] = _params(
        [DIM_VARIANT, DIM_SAMPLE], [np.integer], flag_vars=['is_masked'])

    @classmethod
    def create(cls, data: Any, is_masked: Any = None, **kwargs) -> Dataset:
        arrays = dict(data=data, is_masked=is_masked)
        return _create_dataset(cls, arrays, **kwargs)


class HaplotypeCallDataset(GeneticDataset):

    params: Mapping[Hashable, ArrayParameter] = _params(
        [DIM_VARIANT, DIM_SAMPLE], [np.integer])

    @classmethod
    def create(cls, data: Any, is_phased: Any = None, is_masked: Any = None, **kwargs) -> Dataset:
        arrays = dict(data=data, is_phased=is_phased, is_masked=is_masked)
        return _create_dataset(cls, arrays, **kwargs)


class GenotypeCallDataset(GeneticDataset):

    params: Mapping[Hashable, ArrayParameter] = _params(
        [DIM_VARIANT, DIM_SAMPLE, DIM_PLOIDY], [np.integer])

    @classmethod
    def create(cls, data: Any, is_phased: Any = None, is_masked: Any = None, **kwargs) -> Dataset:
        arrays = dict(data=data, is_phased=is_phased, is_masked=is_masked)
        return _create_dataset(cls, arrays, **kwargs)

    @staticmethod
    @transmutation(HaplotypeCallDataset)
    def to(ds: Dataset, contig: int) -> Dataset:
        """Convert to haplotypecalls"""
        # FIXME: nonsense for testing
        data = _mask(ds.assign(data=ds.data[..., contig]))
        return _transmute(HaplotypeCallDataset.create, ds, data)

    @staticmethod
    @transmutation(GenotypeCountDataset)
    def to(ds: Dataset) -> Dataset:
        """Convert to genotype counts"""
        data = _mask(ds.assign(data=(ds.data > 0).sum(dim=DIM_PLOIDY)))
        return _transmute(GenotypeCountDataset.create, ds, data)


class GenotypeProbabilityDataset(GeneticDataset):

    params: Mapping[Hashable, ArrayParameter] = _params(
        [DIM_VARIANT, DIM_SAMPLE, DIM_PLOIDY, DIM_ALLELE], [np.floating])

    @classmethod
    def create(cls, data: Any, is_phased: Any = None, is_masked: Any = None, **kwargs) -> Dataset:
        arrays = dict(data=data, is_phased=is_phased, is_masked=is_masked)
        return _create_dataset(cls, arrays, **kwargs)

    @staticmethod
    @transmutation(GenotypeDosageDataset)
    def to(ds: Dataset) -> Dataset:
        if not is_shape_match(ds, {DIM_PLOIDY: 2, DIM_ALLELE: 2}):
            raise ValueError(
                'Dosage calculation currently only supported for bi-allelic, '
                'diploid arrays (ploidy and alelle dims must have size 2)')
        # Get array slices for ref and alt probabilities on each chromosome
        c0ref, c1ref = ds.data[..., 0, 0], ds.data[..., 1, 0]
        c0alt, c1alt = ds.data[..., 0, 1], ds.data[..., 1, 1]
        # Compute dosage as float in [0, 2]
        data = c0ref * c1alt + c0alt * c1ref + 2 * c0alt * c1alt
        data = _mask(ds.assign(data=data))
        return _transmute(GenotypeDosageDataset.create, ds, data)


class GenotypeAlleleCountDataset(GeneticDataset):

    params: Mapping[Hashable, ArrayParameter] = _params(
        [DIM_VARIANT, DIM_SAMPLE, DIM_PLOIDY], [np.integer],
        core_vars=['data', 'indexes'], flag_vars=['is_masked'])

    @classmethod
    def create(cls, data: Any, indexes: Any = None, is_masked: Any = None, **kwargs) -> Dataset:
        arrays = dict(data=data, indexes=indexes, is_masked=is_masked)
        return _create_dataset(cls, arrays, **kwargs)

    @classmethod
    def to(cls, ds, dstype):
        return dstype


# ----------------- #
# Factory Functions #
# ----------------- #


for dstype in GeneticDataset.__subclasses__():
    fn_name = to_snake_case(dstype.__name__)
    globals()[f'create_{to_snake_case(dstype.__name__)}'] = dstype.create


# -------------------- #
# Accessor Definitions #
# -------------------- #


@xr.register_dataset_accessor("to")
class DatasetTransmutationAccessor():

    def __init__(self, ds):
        def add_fn(name, fn, ds):
            import functools
            ifn = lambda *args, **kwargs: fn(ds, *args, **kwargs)
            ifn = functools.update_wrapper(ifn, fn)
            setattr(self, name, ifn)
        for dstype, fn in GeneticDataset.transmutations[ds.attrs['type'].__name__].items():
            add_fn(to_snake_case(dstype), fn, ds)
