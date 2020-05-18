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
    threshold: float=0.2,
    window: Optional[int]=None, 
    step: Optional[int]=1, 
    unit: Literal['index', 'physical']='index',
    target_chunk_size: Optional[int]=None, 
    scores=None,
    axis_intervals_kwargs={},
    ld_matrix_kwargs={},
    mis_kwargs={},
    **kwargs
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
    threshold : float
        R2 threshold above which variants are considered in LD.  This should almost
        always be something at least slightly above 0 to avoid the large density very
        near zero LD present in most datasets.  Defaults to 0.2
    window : int
        Size of window. When `unit`='index' this is a fixed window measured in
        rows along the array, e.g. `window`=1 means every row/variant is compared
        only with one other row (immediately below it in the array). When `unit`=
        `physical`, this is measured in physical units according to positions in
        another vector within `ds` (controlled by the config property `variable.pos`,
        typically set as `pos` containing base pair coordinates in a chromosome). 
    step : int, optional
        Fixed step size to move each window by.  Has no effect when `unit`='physical'
        and must be provided when base pair ranges are not being used.  Must
        be less than or equal to `window`.
    unit : str
        Unit type for intervals, one of:
        - 'index': Windows are measured in slices along an array (i.e. "fixed" windows)
        - 'physical': Windows represent ranges within `ds` data variable determined by 
            `config.variable.pos`
    target_chunk_size : Optional[int], optional
        Target size for the total span of adjacent windows, by default None.
        A new chunk is created every time the union of adjacent windows exceeds this
        value.
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
    kwargs:
        'intervals' and 'ld_matrix' can be included as kwargs if predefined

    Returns
    -------
    DataMapping
        Data mapping with single variable `index_to_drop` containing pruned row indexes 
    """
    intervals = kwargs.get(
        'intervals',
        axis_intervals(
            ds, window=window, step=step, unit=unit, 
            target_chunk_size=target_chunk_size, **axis_intervals_kwargs
        )
    )
    ldm = kwargs.get(
        'ld_matrix',
        ld_matrix(
            ds, intervals=intervals, threshold=threshold, 
            scores=scores, **ld_matrix_kwargs
        )
    )
    mis = maximal_independent_set(ldm, **mis_kwargs)
    return mis
