import numba
import numpy as np
from xarray import Dataset
from ...typing import DataMapping
from ...dispatch import NumbaBackend, dispatches_from
from .. import core


@numba.njit(nogil=True)
def __maximal_independent_set(idi, idj, cmp):
    """Numba Sequential greedy maximal independent set implementation 

    Parameters
    ----------
    idi : array-like (M,)
    idj : array-like (M,)
    cmp : array-like (M,) or (0,)

    Returns
    -------
    lost : list[int]
        List of indexes to drop of length <= M
    """
    lost = set()
    assert len(idi) == len(idj)
    
    for k in range(len(idi)):
        i, j = idi[k], idj[k]

        # Only consider upper triangle 
        if j <= i:
            continue

        # Assert presort for unrolled loop
        if k > 0:
            if i < idi[k-1] or (i == idi[k-1] and j < idj[k-1]):
                raise ValueError(f'Edges must be sorted by vertex id')

        # Always ignore dropped vertex in outer loop
        if i in lost:
            continue

        # Decide whether to drop row 
        # cmp = 1 => idi greater, -1 => idj greater
        if len(cmp) > 0 and cmp[k] < 0:
            lost.add(i)
        else:
            lost.add(j)
    return list(lost)


def _maximal_independent_set(idi, idj, cmp=None):
    if cmp is None or len(cmp.shape) == 0 or len(cmp) == 0:
        cmp = np.empty(0, dtype='int8')
    return __maximal_independent_set(idi, idj, cmp)


@dispatches_from(core.maximal_independent_set)
def maximal_independent_set(ds: DataMapping) -> DataMapping:
    """Numba MIS
    
    This method is based on the PLINK algorithm that selects independent 
    vertices from a graph implied by excessive LD between variants.

    For an outline of this process, see [this discussion]
    (https://groups.google.com/forum/#!msg/plink2-users/w5TuZo2fgsQ/WbNnE16_xDIJ).

    Raises
    ------
    ValueError if `i` and `j` are not sorted ascending (and in that order)

    Returns
    -------
    Dataset
    """
    return core._maximal_independent_set(_maximal_independent_set, ds)


core.register_backend_function(NumbaBackend)(maximal_independent_set)