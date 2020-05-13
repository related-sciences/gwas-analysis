from ..dispatch import register_function, register_backend_function
from xarray import Dataset
from typing import Optional
from typing_extensions import Literal
from ..stats.core import axis_intervals, ld_matrix
from ..graph.core import maximal_independent_set
from ..typing import DataMapping
from . import DOMAIN

register_backend_function = register_backend_function(DOMAIN)


def ld_prune(
    ds: Dataset,
    window: Optional[int]=None, 
    step: Optional[int]=1, 
    unit: Literal['index', 'physical']='index',
    target_chunk_size: Optional[int]=None, 
    threshold: Optional[float]=0.2,
    scores=None,
    axis_intervals_kwargs={},
    ld_matrix_kwargs={},
    mis_kwargs={}
) -> DataMapping:
    """LD Prune 

    Prune variants within a dataset using a sparse LD matrix to find a 
    maximally independent set (MIS).

    Note: This result is not a true MIS if `scores` are provided and are based on MAF 
    or anything else that is not identical for all variants.

    Parameters
    ----------
    ds : Dataset
        Dataset transmutable to `GenotypeCountDataset`
    window : int
        Size of window. This is either in units implied by `positions` 
        or is a fixed window along the axis otherwise.  For example,
        a fixed size window of 1 for row variants corresponds to intervals
        containing only one variant and its neighbor. 
    step : int, optional
        Fixed step size to move each window by.  Has no effect when `positions` is provided
        and must be provided when base pair ranges are not being used.  Must
        be less than or equal to `window`.
    unit : str
        Unit type for intervals, one of:
        - 'index': Windows are measured in slices along an array (i.e. "fixed" windows)
        - 'physical': Windows represent ranges within `ds` data variable determined by 
            `config.variable.positions`, e.g. 'pos' is often the variable within `ds` 
            that represents genomic positions
    target_chunk_size : Optional[int], optional
        Target size for the total span (i.e.) of adjacent windows, by default None.
        A new chunk is created every time the union of adjacent windows exceeds this
        value.
    threshold : float, optional
        R2 threshold above which variants are considered in LD.  This should almost
        always be something at least slightly above 0 to avoid the large density very
        near zero LD present in most datasets.  Defaults to 0.2
    scores : str or array-like, optional
        Name of field to use to prioritize variant selection (e.g. MAF).  Will
        be used directly if provided as an array and otherwise fetched from 
        `ds` if given as a variable name.
    axis_intervals_kwargs: 
        Extra arguments for `axis_intervals`
    ld_matrix_kwargs: 
        Extra arguments for `ld_matrix`
    mis_kwargs: 
        Extra arguments for `maximal_independent_set`

    Returns
    -------
    DataMapping
        Data mapping with single variable `index_to_drop` containing pruned row indexes 
    """
    intervals = axis_intervals(
        ds, window=window, step=step, unit=unit, 
        target_chunk_size=target_chunk_size, **axis_intervals_kwargs
    )
    ldm = ld_matrix(
        ds, intervals=intervals, threshold=threshold, scores=scores, **ld_matrix_kwargs
    )
    mis = maximal_independent_set(ldm, **mis_kwargs)
    return mis
