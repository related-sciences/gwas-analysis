import numba
import numpy as np
from ..dispatch import register_backend
from .maximal_independent_set import MISBackend
from .core import DOMAIN


@numba.njit
def _maximal_independent_set(idi, idj, cmp):
    """Sequential greedy maximal independent set implementation 
    
    This is based on the PLINK algorithm that selects independent vertices from
    a graph typically implied by excessive LD between variants.

    For an outline of this process, see https://groups.google.com/forum/#!msg/plink2-users/w5TuZo2fgsQ/WbNnE16_xDIJ

    Parameters
    ----------
    idi : array-like (M,)
        Sequence of row indexes in the upper triangle of a square matrix
    idj : array-like (M,)
        Sequence of column indexes in the upper triangle of a square matrix
    cmp : array-like (M,) or (0,)
        Sequence of comparator values corresponding to whether or not the 
        left side of a pair (`idi`) is greater than the right side (`idj`)
        according to some external value (typically MAF).
        Values: 
            >0 - The left side is greater
            <0 - The right side is greater
            0  - The two are equal (lowest index is kept)

    Raises
    ------
    ValueError if `idi` and `idj` are not sorted ascending (and in that order)

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

        # Always ignore dropped vertex in slowest chan
        if i in lost:
            continue

        # Decide whether to drop row 
        # cmp = 1 => idi greater, -1 => idj greater
        if len(cmp) > 0 and cmp[k] < 0:
            lost.add(i)
        else:
            lost.add(j)
    return list(lost)


def maximal_independent_set(idi, idj, cmp=None):
    if cmp is None or len(cmp) == 0:
        cmp = np.empty(0, dtype='int8')
    return _maximal_independent_set(idi, idj, cmp)


@register_backend(DOMAIN)
class NumbaBackend(MISBackend):

    id = 'numba'

    def run(self, *args, **kwargs):
        return maximal_independent_set(*args, **kwargs)
