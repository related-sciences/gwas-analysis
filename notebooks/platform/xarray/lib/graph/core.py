"""Graph API"""
from ..dispatch import register_function, register_backend_function
from xarray import Dataset
import numpy as np
from ..typing import DataMapping
from . import DOMAIN

register_backend_function = register_backend_function(DOMAIN)

def wrap_mis_vec_fn(fn):
    def mis(ds, *args, **kwargs):
        args = [np.asarray(ds[c]) for c in ['i', 'j']]
        if 'cmp' in ds:
            args.append(ds['cmp'])
        drop = fn(*args, **kwargs)
        return Dataset({'index_to_drop': ('index', np.array(drop))})
    return mis

@register_function(DOMAIN)
def maximal_independent_set(ds: DataMapping) -> Dataset:
    """Maximal Independent Set
    
    Parameters
    ----------
    ds : DataMapping
        Mapping with vars 'i', 'j', and 'cmp' ('cmp' optional).
        Rows should indicate edges with `i` and `j` as vertices.
        If present, `cmp` is used to prioritize one vertex over 
        another and  should be equal to a number with values that 
        correspond to a typical comparator result:
            >0 - The left side is greater
            <0 - The right side is greater
            0  - The two are equal (lowest index is kept)
    
    Returns
    -------
    Dataset
        A dataset with single variable 'index_to_drop'
    """
    pass

