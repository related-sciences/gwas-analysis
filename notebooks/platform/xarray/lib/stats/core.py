from ..dispatch import register_function
from xarray import Dataset
from typing import Optional

DOMAIN = 'stats'

@register_function(DOMAIN)
def ld_matrix(
    ds: Dataset,
    window: int = 1_000_000, 
    threshold: float = 0.2,
    step: Optional[int] = None, 
    groups='contig', 
    positions='pos', 
    scores=None,
    **kwargs
):
    """Compute Sparse LD Matrix (for tall-skinny matrices only at the moment)

    This method works by first computing all overlapping variant ranges 
    and then breaking up necessary pariwise comparisons into approximately 
    equal sized chunks (along dimension 0).  These chunks are exactly equal 
    in size when windows are of fixed size but vary based on variant density 
    when using base pair ranges. For each pair of variants in a window,
    R2 is calculated and only those exceeding the provided threshold are returned.

    Parameters
    ----------
    window : int
        Size of window for LD comparisons (between rows).  This is either in base pairs 
        if `positions` is not None or is a fixed window size otherwise.  By default this
        is 1,000,000 (1,000 kbp)
    threshold : float
        R2 threshold below which no variant pairs will be returned.  This should almost
        always be something at least slightly above 0 to avoid the large density very
        near zero LD present in most datasets.  Defaults to 0.2
    step : optional
        Fixed step size to move each window by.  Has no effect when `positions` is provided
        and must be provided when base pair ranges are not being used.
    groups : str or array-like, optional
        Name of field to use to represent disconnected components (typically contigs).  Will
        be used directly if provided as an array and otherwise fetched from
        `ds` if given as a variable name.
    positions : str or array-like, optional
        Name of field to use to represent base pair positions.  Will
        be used directly if provided as an array and otherwise fetched from 
        `ds` if given as a variable name.
    scores : [type], optional
        Name of field to use to prioritize variant selection (e.g. MAF).  Will
        be used directly if provided as an array and otherwise fetched from 
        `ds` if given as a variable name.
    return_intervals : bool
        Whether or not to also return the variant interval calculations (which can be 
        useful for analyzing variant density), by default False
    **kwargs
        Backend-specific options

    Returns
    -------
    DataFrame or (DataFrame, (DataFrame, DataFrame))
        Upper triangle (including diagonal) of LD matrix as COO in dataframe.  Fields:
        `i`: Row (variant) index 1
        `j`: Row (variant) index 2
        `value`: R2 value
        `cmp`: If scores are provided, this is 1, 0, or -1 indicating whether or not
            i > j (1), i < j (-1), or i == j (0)
        When `return_intervals` is True, the second tuple contains the results from
        `axis_intervals`
    """
    pass
