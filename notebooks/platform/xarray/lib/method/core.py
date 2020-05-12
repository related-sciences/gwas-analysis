from ..dispatch import register_function, register_backend_function
from xarray import Dataset
from typing import Optional
from ..typing import DataFrame
from . import DOMAIN

register_backend_function = register_backend_function(DOMAIN)

@register_function(DOMAIN)
def ld_prune(
    ldm: DataFrame,
    use_cmp: bool = True
) -> DataFrame:
    """LD Prune 

    Prune variants within a dataset using a sparse LD matrix to find a 
    maximally independent set (MIS).

    Note: This result is not a true MIS if `use_cmp` is True and was based on MAF scores 
    (or anything else) provided during pair-wise LD matrix evaluation, or if those scores 
    were not all identical (it is otherwise).  

    Parameters
    ----------
    ldm : DataFrame
        LD matrix from `ld_matrix`
    use_cmp : bool
        Whether or not to use precomputed score-based comparisons

    Returns
    -------
    DataFrame
        Dataframe with single field 'index_to_drop' containing pruned row indexes 
    """
    pass