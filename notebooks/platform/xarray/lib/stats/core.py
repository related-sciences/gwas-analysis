import xarray as xr
import pandas as pd
from xarray import Dataset, DataArray
from typing import Optional, Union, Type, Tuple, Callable
from typing_extensions import Literal
from numpy import ndarray
from ..dispatch import register_function, register_backend_function
from ..typing import DataFrame, BlockArray
from . import DOMAIN

register_backend_function = register_backend_function(DOMAIN)

@register_function(DOMAIN)
def ld_matrix(
    ds: Dataset,
    intervals: Optional[Tuple[DataArray, DataArray]]=None,
    threshold: Optional[float]=None,
    scores=None,
    **kwargs
) -> DataFrame:
    """Compute Sparse LD Matrix (for tall-skinny matrices only at the moment)

    This method works by first computing all overlapping variant ranges 
    and then breaking up necessary pariwise comparisons into approximately 
    equal sized chunks (along dimension 0).  These chunks are exactly equal 
    in size when windows are of fixed size but vary based on variant density 
    when using base pair ranges. For each pair of variants in a window,
    R2 is calculated and only those exceeding the provided threshold are returned.

    Parameters
    ----------
    ds : Dataset
        Dataset transmutable to `GenotypeCountDataset`
    intervals: Tuple[DataArray, DataArray], optional
        Results from `axis_intervals` used to limit LD computations, by default None
        implying that the full LD matrix should be computed.
    threshold : float, optional
        R2 threshold below which no variant pairs will be returned.  This should almost
        always be something at least slightly above 0 to avoid the large density very
        near zero LD present in most datasets.  Defaults to None
    scores : str or array-like, optional
        Name of field to use to prioritize variant selection (e.g. MAF).  Will
        be used directly if provided as an array and otherwise fetched from 
        `ds` if given as a variable name.
    **kwargs
        Backend-specific options

    Returns
    -------
    DataFrame
        Upper triangle (including diagonal) of LD matrix as COO in dataframe.  Fields:
        `i`: Row (variant) index 1
        `j`: Row (variant) index 2
        `value`: R2 value
        `cmp`: If scores are provided, this is 1, 0, or -1 indicating whether or not
            i > j (1), i < j (-1), or i == j (0)
    """
    pass


@register_function(DOMAIN)
def _ld_matrix_block(
    x: BlockArray,
    intervals: Tuple[BlockArray, BlockArray],
    threshold: Optional[float]=None,
    scores: Optional[BlockArray]=None,
    return_value: bool=True, 
    **kwargs
) -> DataFrame:
    """Compute LD Matrix for block

    Parameters
    ----------
    x : BlockArray (M, N)
        Data array for block
    intervals : Tuple[BlockArray, BlockArray]
        `(axis_intervals, chunk_interval)` where:
        - `axis_intervals` contains intervals for rows within block
        - `chunk_interval` contains summarization of interval data for single block
    threshold : float, optional
        Minimum R2 threshold below which no results will be returned, by default None
    scores : BlockArray (M,), optional
        Array of scores, by default None
    return_value : bool
        Whether or not to return the R2 `value` field.
        This is useful in combination with a `threshold` to establish only
        links between rows/variants with a minor decrease in storage necessary
        for the `value` float array in results.
    kwargs:
        Backend-specific arguments

    Returns
    -------
    DataFrame
        Same as `ld_matrix`
    """
    pass

@register_function(DOMAIN)
def axis_intervals(
    ds: Dataset,
    window: Optional[int]=None, 
    step: Optional[int]=1, 
    unit: Literal['index', 'physical']='index',
    target_chunk_size: Optional[int]=None, 
    dtype: Union[str, Type]='int32',
    **kwargs
) -> Tuple[DataArray, DataArray]:
    """Axis interval calculations for overlapping block-wise computations

    This function can be used to generate the following for any axis of an array,
    though this is typically used for rows representing variants:

    1. Which items are in the same window using a window and step size between windows
    2. Which items are in the same window using a measure of physical distance (usually 
    genomic distance)

    The calculation operates by looping sequentially for `n` iterates, either passed
    directly or inferred from other vectors, and identifying items in windows while
    also generating chunk information.  This chunk information is crucial for 
    effectively sizing overlapping blocks and is controlled by the `target_chunk_size`
    parameter.  

    Parameters
    ----------
    ds : Dataset
        Dataset containing vectors to use for interval calculations.  
    window : int
        Size of window. When `unit`='index' this is a fixed window measured in
        rows along the array, e.g. `window`=1 means every row/variant is compared
        only with one other row (immediately below it in the array). When `unit`=
        `physical`, this is measured in physical units according to positions in
        another vector within `ds` (controlled by `config.variable.pos`,
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
        Target size for the total span (i.e.) of adjacent windows, by default None.
        A new chunk is created every time the union of adjacent windows exceeds this
        value.
    dtype : Union[str, Type], optional
        Type to use for axis index values in result, by default np.int32
    kwargs:
        Backend-specific arguments
        
    Returns
    -------
    Tuple[DataArray, DataArray]
        Tuple with axis and chunk interval definitions
    """
    pass


def impute_mean(ds: Dataset, dim: str, variable='data'):
    """Mean impute variable in a dataset
    
    Parameters
    ----------
    ds : Dataset
        Dataset with some variable to impute.
        When missing values are present, this **must** contain
        an `is_masked` variable (otherwise nothing will 
        happen).
    dim : str
        Dimension over which means should be computed.
        For example, in a (variants, samples) data array,
        `dim='variants'` means that means are computed
        for each sample and missing values are replaced
        with that associated mean (per-variant).
    variable : str
        Variable in `ds` to impute

    Returns
    -------
    Dataset
        Dataset with `variable` imputed and `is_masked` dropped.  
        Note that this often leads to a type change (e.g. int8 -> float64)
    """
    if not 'is_masked' in ds:
        return ds
    return (
        ds
        .assign(**{variable: lambda ds: xr.where(
            ds.is_masked, 
            ds[variable].mean(dim=dim), 
            ds[variable]
        )})
        .drop('is_masked')
    )


def _ld_matrix_args(    
    ds: Dataset,
    intervals: Optional[Tuple[DataArray, DataArray]]=None,
    scores=None
):
    """Default `ld_matrix` argument processor"""
    # This will coerce any dataset provided to the correct format
    # (if the coercion is supported by whatever dataset type was given)
    ds = ds.to.genotype_count_dataset()

    # 2D array of allele counts
    x = ds.data

    # Compute intervals for full LD matrix by default
    if intervals is None:
        intervals = axis_intervals(ds, window=None, step=1, unit='index')
    ais, _ = intervals

    if x.shape[0] < ais.shape[0]:
        raise ValueError(
            f'Data array with shape {x.shape} cannot have fewer '
            'rows than axis intervals with shape {ais.shape}'
        )
    if scores is not None and x.shape[0] != scores.shape[0]:
        raise ValueError(
            f'Data array with shape {x.shape} must have same number '
            'of rows as scores with shape {scores.shape}'
        )

    return x, intervals


def _ld_matrix_block_impl(
    fn: Callable,
    x: BlockArray, 
    intervals: Tuple[BlockArray, BlockArray], 
    threshold: Optional[float]=None,
    scores: Optional[BlockArray]=None, 
    return_value: bool = True,
    **kwargs
):
    """Default LD matrix block processor"""
    axis_intervals, chunk_interval = intervals
    assert chunk_interval.ndim == 1
    assert x.shape[0] >= axis_intervals.shape[0]
    if scores is not None:
        assert x.shape[0] == scores.shape[0]

    # Get vector results from delegate
    idx, res, cmp = fn(
        x, 
        axis_intervals, 
        chunk_interval, 
        scores, 
        **kwargs
    )

    # Subset vectors if necessary
    if threshold is not None:
        mask = res >= threshold
        idx, res = idx[mask], res[mask]
        if scores is not None:
            cmp = cmp[mask]

    # Build DF result with minimum fields necessary
    cols = dict(i=idx[:, 0], j=idx[:, 1])
    if return_value:
        cols['value'] = res
    if scores is not None:
        cols['cmp'] = cmp
    df = pd.DataFrame(cols)
    return df

        