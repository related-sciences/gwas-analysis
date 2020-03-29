from __future__ import annotations
import xarray as xr
import numpy as np
from lib import DIM_VARIANT, DIM_SAMPLE, DIM_PLOIDY, DIM_ALLELE
from lib.utils import is_array, try_cast_unsigned, check_array, check_domain, is_signature_compatible
from typing import Union, Sequence, Any

class ShapeError(Exception):
    pass
    
class Array(xr.DataArray):
    __slots__ = []
    
    @staticmethod
    def dispatch(cls, *args, **kwargs):
        if is_signature_compatible(cls.__init__, None, *args, **kwargs):
            return super().__new__(cls)
        return xr.DataArray(*args, **kwargs)

class Dataset(xr.Dataset):
    __slots__ = []
    
    @staticmethod
    def dispatch(cls, *args, **kwargs):
        if is_signature_compatible(cls.__init__, None, *args, **kwargs):
            return super().__new__(cls, *args, **kwargs)
        return xr.Dataset(*args, **kwargs)


def coerce_array_arg(fn, args):
    if args and is_array(args[0]):
        args = (fn(args[0]),) + args[1:]
    return args

def array_kwargs(data, kwargs, dims=None, set_coords=True):
    if 'dims' not in kwargs and dims:
        kwargs['dims'] = dims
    if 'coords' not in kwargs and dims and set_coords:
        kwargs['coords'] = {dims[i]: np.arange(data.shape[i]) for i in range(data.ndim)}
    return kwargs
        
    
# # (variant, sample) -> dosage
# # floats interpreted as probability dosages
# # ints interpreted as alt allele counts
class GenotypeDosageArray(Array):
    DIMS = [DIM_VARIANT, DIM_SAMPLE]
    __slots__ = []
    
    def __new__(cls, *args, **kwargs):
        return cls.dispatch(cls, *args, **kwargs)

    def __init__(self, *, allele_dosage: Any, **kwargs):
        data = allele_dosage
        data = check_array(self.__class__, data, types=[np.unsignedinteger, np.floating], dims=self.DIMS)
        super().__init__(data, **array_kwargs(data, kwargs, self.DIMS))
        
# (variant, sample, ploidy) -> allele index
class GenotypeIndexArray(Array):
    DIMS = [DIM_VARIANT, DIM_SAMPLE, DIM_PLOIDY]
    __slots__ = []
    
    def __new__(cls, *args, **kwargs):
        return cls.dispatch(cls, *args, **kwargs)

    def __init__(self, *, allele_index: Any, **kwargs):
        data = allele_index
        data = try_cast_unsigned(data)
        data = check_array(self.__class__, data, types=[np.unsignedinteger], dims=self.DIMS)
        super().__init__(data, **array_kwargs(data, kwargs, self.DIMS))
        
    def to_dosage_array(self) -> GenotypeDosageArray:
        return GenotypeDosageArray(allele_dosage=self.sum(dim='ploidy').data, attrs=self.attrs)
        
# (variant, sample, ploidy, allele) -> allele probability
class GenotypeProbabilityArray(Array): 
    DIMS = [DIM_VARIANT, DIM_SAMPLE, DIM_PLOIDY, DIM_ALLELE]
    __slots__ = []
    
    def __new__(cls, *args, **kwargs):
        return cls.dispatch(cls, *args, **kwargs)

    def __init__(self, *, allele_probability: Any, **kwargs):
        data = allele_probability
        data = check_array(self.__class__, data, types=[np.floating], dims=self.DIMS)
        data = check_domain(self.__class__, data, 0, 1)
        super().__init__(data, **array_kwargs(data, kwargs, self.DIMS))
        
    def to_dosage_array(self) -> GenotypeDosageArray:
        if (self.sizes['ploidy'] != 2 or self.sizes['allele'] != 2):
            raise ValueError(
                'Dosage calculation currently only supported for bi-allelic, '
                'diploid arrays (ploidy and alelle dims must have size 2)')
        # Get array slices for ref and alt probabilities on each chromosome
        c0ref, c1ref = self[..., 0, 0], self[..., 1, 0] 
        c0alt, c1alt = self[..., 0, 1], self[..., 1, 1] 
        # Compute dosage as float in [0, 2]
        dosage = c0ref * c1alt + c0alt * c1ref + 2 * c0alt * c1alt
        return GenotypeDosageArray(allele_dosage=dosage, attrs=self.attrs)
        
# (variant, sample, ploidy) -> allele count
class GenotypeCountArray(Array):
    DIMS = [DIM_VARIANT, DIM_SAMPLE, DIM_PLOIDY]
    __slots__ = []
    
    def __new__(cls, *args, **kwargs):
        return cls.dispatch(cls, *args, **kwargs)

    def __init__(self, *, allele_count: Any, **kwargs):
        data = allele_count
        data = try_cast_unsigned(data)
        data = check_array(self.__class__, data, types=[np.unsignedinteger], dims=self.DIMS)
        super().__init__(data, **array_kwargs(data, kwargs, self.DIMS))

        
# # (variant, sample, ploidy) -> allele index
# # (variant, sample, ploidy) -> allele count
class GenotypeAlleleCountDataset(Dataset):
    __slots__ = []
    
    def __new__(cls, *args, **kwargs):
        return cls.dispatch(cls, *args, **kwargs)
        
    def __init__(self, *, allele_index: GenotypeIndexArray, allele_count: GenotypeCountArray, **kwargs):
        data_vars = {
            'allele_index': allele_index,
            'allele_count': allele_count
        }
        super().__init__(data_vars, **kwargs)
