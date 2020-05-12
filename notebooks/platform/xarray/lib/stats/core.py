from ..dispatch import register_function, register_backend_function
from xarray import Dataset, DataArray
from typing import Optional, Union, Type, Tuple
from ..typing import DataFrame
from typing_extensions import Literal
from . import DOMAIN

register_backend_function = register_backend_function(DOMAIN)

@register_function(DOMAIN)
def ld_matrix(
    ds: Dataset,
    intervals: Optional[Tuple[Dataset, Dataset]]=None,
    threshold: Optional[float]=0.2,
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
    intervals: Tuple[Dataset, Dataset], optional
        Results from `axis_intervals` used to limit LD computations, by default None
        implying that the full LD matrix should be computed.
    threshold : float, optional
        R2 threshold below which no variant pairs will be returned.  This should almost
        always be something at least slightly above 0 to avoid the large density very
        near zero LD present in most datasets.  Defaults to 0.2
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
def axis_intervals(
    ds: Dataset,
    window: Optional[int]=None, 
    step: Optional[int]=1, 
    unit: Literal['index', 'physical']='index',
    target_chunk_size: Optional[int]=None, 
    dtype: Union[str, Type]='int32'
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
    dtype : Union[str, Type], optional
        Type to use for axis index values in result, by default np.int32

    Returns
    -------
    Tuple[DataArray, DataArray]
        Tuple with axis and chunk interval definitions
    """
    pass
