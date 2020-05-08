import numpy as np
import dask
import dask.array as da
from typing import Union

T_ARRAY = Union[dask.array.Array, np.ndarray]


def gramian(a: T_ARRAY) -> T_ARRAY:
    """Returns gramian matrix of the given matrix"""
    return da.dot(a.T, a)


def __clip_mu_maf(mu: T_ARRAY, maf: float) -> T_ARRAY:
    # TODO (rav): why  mu[(mu <= maf) & (mu >= 1 - maf)] doesnt work, do I need to use
    #             bitwise_and (isn't it the same?)
    mu[mu >= 1.0 - maf] = 1.0
    mu[mu <= maf] = 1.0
    return mu


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
    # 𝔼[gs|V] = 1β0 + Vβ, where 1 is a length _s_ vector of 1s, and β = (β1,...,βD)^T
    # is a length D vector of regression coefficients for each of the PCs
    # TODO: rechunk needs to go
    pcsi = da.concatenate([da.from_array(np.ones((1, pcs.shape[1]))), pcs], axis=0).rechunk()
    q, r = da.linalg.qr(pcsi.T)
    # mu, eq: 3
    half_beta = da.matmul(da.matmul(da.linalg.inv(2 * r), q.T), g.T)
    mu = da.matmul(pcsi.T, half_beta).T
    # phi, eq: 4
    clipped_mu = mu.map_blocks(lambda i: __clip_mu_maf(i, maf))
    variance = clipped_mu.map_blocks(lambda i: i * (1.0 - i))
    stddev = da.sqrt(variance)
    centered_af = g / 2 - clipped_mu
    # correction for clipped mu
    centered_af[clipped_mu == 1.0] = 0.0
    return gramian(centered_af) / gramian(stddev)
