import numpy as np
import dask
import dask.array as da
from typing import Union

T_ARRAY = Union[dask.array.Array, np.ndarray]


def gramian(a: T_ARRAY) -> T_ARRAY:
    """Returns gramian matrix of the given matrix"""
    return a.T.dot(a)


def impute_with_variant_mean(g: T_ARRAY) -> (T_ARRAY, T_ARRAY):
    """
    Imputes missing values (negative or np.nan) with variant mean.

    :param g: variants x samples matrix, shape (v x s)
    :returns: tuple of (missing_values_mask, imputed_g)
    """
    neg_values_are_nan = da.where(g >= 0.0, g, np.nan)
    is_nan_mask = da.isnan(neg_values_are_nan)
    variant_mean = da.ma.masked_array(g, mask=is_nan_mask).mean(axis=1)[:, np.newaxis]
    return is_nan_mask, da.where(is_nan_mask, variant_mean, g)


# TODO (rav):
#  * double check MAF range (0.0, 1.0) or [0.0, 0.5]
#  * add support for sample.include
def pc_relate(pcs: T_ARRAY,
              g: T_ARRAY,
              maf: float = 0.01) -> T_ARRAY:
    """
    Compute PC-Relate https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4716688/

    :param pcs: PCs that best capture population structure within g,
                it has a length D
    :param g: variants x samples matrix, shape (v x s)
    :param maf: individual minor allele frequency filter. If an individual's estimated
                individual-specific minor allele frequency at a SNP is less than this value,
                that SNP will be excluded from the analysis for that individual.
                The default value is 0.01. Must be between (0.0, 0.1).
    :returns: (s x s) pairwise recent kinship estimation matrix
    """
    if maf < 0.0 or maf > 1.0:
        raise ValueError("MAF must be between (0.0, 1.0)")
    missing_g_mask, imputed_g = impute_with_variant_mean(g)
    # 𝔼[gs|V] = 1β0 + Vβ, where 1 is a length _s_ vector of 1s, and β = (β1,...,βD)^T
    # is a length D vector of regression coefficients for each of the PCs
    # TODO: rechunk needs to go
    pcsi = da.concatenate([da.from_array(np.ones((1, pcs.shape[1]))), pcs], axis=0).rechunk()
    # TODO (rav): qr is likely not going to scale given the requirements of the current
    #             implementation
    q, r = da.linalg.qr(pcsi.T)
    # mu, eq: 3
    half_beta = da.linalg.inv(2 * r).dot(q.T).dot(imputed_g.T)
    mu = pcsi.T.dot(half_beta).T
    # phi, eq: 4
    mask = (mu <= maf) | (mu >= 1.0 - maf) | missing_g_mask
    mu_mask = da.ma.masked_array(mu, mask=mask)
    variance = mu_mask.map_blocks(lambda i: i * (1.0 - i))
    variance = da.ma.filled(variance, fill_value=0.0)
    stddev = da.sqrt(variance)
    centered_af = g / 2 - mu_mask
    centered_af = da.ma.filled(centered_af, fill_value=0.0)
    return gramian(centered_af) / gramian(stddev)
