from __future__ import annotations
import xarray as xr
import numpy as np
from lib import DIM_VARIANT, DIM_SAMPLE, DIM_PLOIDY, DIM_ALLELE
from lib.utils import is_array, check_array, check_domain, is_shape_match, is_signature_compatible
from lib.ops import get_mask_array, get_filled_array
from typing import Union, Sequence, Any
import collections
       
# TODO:
# - Is there a type hint compatible with duck arrays?
#   - Xarray uses Any (e.g. https://github.com/pydata/xarray/blob/master/xarray/core/dataarray.py#L559)
# - What is the best way to implement code dependent on optional dependencies (i.e. avoid requiring global dask imports)?
# - Are we sure it's worth assuming no support for masking in Xarray?
#   - https://github.com/pydata/xarray/issues/1194

# Notes:
# - xr.DataArray(xr.DataArray()) will call .asarray on input argument pushing .data into memory as numpy
# - Xarray often invokes constructors on DataArray/Dataset like this: ```type(self)(*args, **kwargs)```
#   - This means that subclasses must be able to differentiate between constructor params

MISSING = -1

class Dataset(xr.Dataset):
    __slots__ = []
    
    def __new__(cls, *args, **kwargs):
        if args and isinstance(args[0], collections.Mapping) or 'data_vars' in kwargs:
            return xr.Dataset(*args, **kwargs)    
        return super().__new__(cls, *args, **kwargs)

    def apply_mask(self, data, fill_value=MISSING):
        if 'mask' not in self:
            return data
        return xr.where(self.mask, fill_value, data)

def extract_array(cls, data, types, dims, **kwargs):
    data = check_array(cls, data, types=types, dims=dims)
    mask = kwargs.get('vars', {}).get('mask')
    if mask is None:
        mask = get_mask_array(data)
        if mask is not None:
            data = get_filled_array(data, fill_value=MISSING)
    return data, mask

def get_data_array(data, dims=None, set_coords=True, **kwargs):
    if isinstance(data, xr.DataArray):
        return data
    kwargs = dict(kwargs)
    if 'dims' not in kwargs and dims:
        kwargs['dims'] = dims
    if 'coords' not in kwargs and dims and set_coords:
        kwargs['coords'] = {dims[i]: np.arange(data.shape[i]) for i in range(data.ndim)}
    return xr.DataArray(data, **kwargs)
    
def get_data_vars(arrays, dims=None, **kwargs):
    return {k: get_data_array(v, dims=dims, **kwargs) for k, v in arrays.items() if v is not None}
        
def get_dataset_args(cls, arrays, types, dims, **kwargs):
    arrays = {k: v for k, v in arrays.items() if v is not None}
    arrays['data'], arrays['mask'] = extract_array(cls, arrays['data'], types=types, dims=dims, **kwargs)
    args = [get_data_vars(arrays, dims=dims, **kwargs)]
    kwargs = {'attrs': kwargs.get('attrs')}
    return args, kwargs

# (variant, sample) -> dosage
class GenotypeDosageDataset(Dataset):
    DIMS = [DIM_VARIANT, DIM_SAMPLE]
    __slots__ = []
    
    def __init__(self, data: Any, **kwargs):
        args, kwargs = get_dataset_args(type(self), dict(data=data), types=np.floating, dims=self.DIMS, **kwargs)
        super().__init__(*args, **kwargs)


# (variant, sample) -> alt allele count
class GenotypeCountDataset(Dataset):
    DIMS = [DIM_VARIANT, DIM_SAMPLE]
    __slots__ = []
    
    def __init__(self, data: Any, phased: Any = None, **kwargs):
        arrays = dict(data=data, phased=phased)
        args, kwargs = get_dataset_args(type(self), arrays, types=np.integer, dims=self.DIMS, **kwargs)
        super().__init__(*args, **kwargs)
        
# (variant, sample, ploidy) -> allele index
class GenotypeCallDataset(Dataset):
    DIMS = [DIM_VARIANT, DIM_SAMPLE, DIM_PLOIDY]
    __slots__ = []
    
    def __init__(self, data: Any, phased: Any = None, **kwargs):
        arrays = dict(data=data, phased=phased)
        args, kwargs = get_dataset_args(type(self), arrays, types=np.integer, dims=self.DIMS, **kwargs)
        super().__init__(*args, **kwargs)
        
    def to_count_dataset(self) -> GenotypeCountDataset:
        data = self.apply_mask((self.data > 0).sum(dim=DIM_PLOIDY))
        return GenotypeCountDataset(data, phased=self.get('phased'), attrs=self.attrs, vars=dict(mask=self.get('mask')))
    
# (variant, sample) -> allele index
class HaplotypeCallDataset(Dataset):
    DIMS = [DIM_VARIANT, DIM_SAMPLE]
    __slots__ = []
    
    def __init__(self, data: Any, **kwargs):
        args, kwargs = get_dataset_args(type(self), dict(data=data), types=np.integer, dims=self.DIMS, **kwargs)
        super().__init__(*args, **kwargs)
        
# (variant, sample, ploidy, allele) -> allele probability
class GenotypeProbabilityDataset(Dataset): 
    DIMS = [DIM_VARIANT, DIM_SAMPLE, DIM_PLOIDY, DIM_ALLELE]
    __slots__ = []
    
    def __init__(self, data: Any, phased: Any = None, **kwargs):
        data = check_domain(type(self), data, 0, 1)
        arrays = dict(data=data, phased=phased)
        args, kwargs = get_dataset_args(type(self), arrays, types=np.floating, dims=self.DIMS, **kwargs)
        super().__init__(*args, **kwargs)
        
    def to_dosage_dataset(self) -> GenotypeDosageDataset:
        if not is_shape_match(self, {DIM_PLOIDY: 2, DIM_ALLELE: 2}):
            raise ValueError(
                'Dosage calculation currently only supported for bi-allelic, '
                'diploid arrays (ploidy and alelle dims must have size 2)')
        # Get array slices for ref and alt probabilities on each chromosome
        c0ref, c1ref = self.data[..., 0, 0], self.data[..., 1, 0] 
        c0alt, c1alt = self.data[..., 0, 1], self.data[..., 1, 1] 
        # Compute dosage as float in [0, 2]
        data = c0ref * c1alt + c0alt * c1ref + 2 * c0alt * c1alt
        data = self.apply_mask(data)
        return GenotypeDosageDataset(data, phased=self.get('phased'), attrs=self.attrs, vars=dict(mask=self.get('mask')))

        
# (variant, sample, ploidy) -> allele index
# (variant, sample, ploidy) -> allele count
class GenotypeAlleleCountDataset(Dataset):
    DIMS = [DIM_VARIANT, DIM_SAMPLE, DIM_PLOIDY]
    __slots__ = []
        
    def __init__(self, data: Any, indexes: Any, **kwargs):
        arrays = dict(data=data, indexes=indexes)
        args, kwargs = get_dataset_args(type(self), arrays, types=np.integer, dims=self.DIMS, **kwargs)
        super().__init__(*args, **kwargs)
