# Run various Dask profiling tools (for the local scheduler) on the Dask PC-Relate implementation

# High-level results
# For nvariants = 2,000, nsamples = 10,000 PC-Relate takes about 10s on my 4 core machine
# Of that, around 8s is computing the gramian matrices, an outer matrix multiplication.
# Shapes: (10,000, 2,000) x (2,000, 10,000) = (10,000, 10,000).
# For comparison, numpy takes about the same amount of time to do this multiplication.

# See profiles in pc_relate_profile.html and pc_relate_no_gramian_profile.html

from typing import Union

import dask
import dask.array as da
from dask_ml.decomposition import PCA
import numpy as np

from dask.diagnostics import Profiler, ResourceProfiler, CacheProfiler
from dask.callbacks import Callback

T_ARRAY = Union[dask.array.Array, np.ndarray]

from lib.method.pc_relate.pc_relate import pc_relate, impute_with_variant_mean

def compute_pca(g, display_pc_12: bool=False):
    pca = PCA(n_components=8, random_state=42)
    pca.fit(g)
    if display_pc_12:
        display(sns.scatterplot(x=pca.components_[0], y=pca.components_[1]))
    pcs = da.from_array(pca.components_)
    return pcs[:2,:]

def pc_relate_no_gramian(pcs: T_ARRAY, g: T_ARRAY, maf: float = 0.01) -> T_ARRAY:
    if maf <= 0.0 or maf >= 1.0:
        raise ValueError("MAF must be between (0.0, 1.0)")
    missing_g_mask, imputed_g = impute_with_variant_mean(g)
    # 𝔼[gs|V] = 1β0 + Vβ, where 1 is a length _s_ vector of 1s, and β = (β1,...,βD)^T
    # is a length D vector of regression coefficients for each of the PCs
    # TODO: rechunk needs to go
    pcsi = da.concatenate(
        [da.from_array(np.ones((1, pcs.shape[1]))), pcs], axis=0
    ).rechunk()
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
    return centered_af, stddev # no gramian call here

def printkeys(key, dask, state):
    """Dask profiling helper"""
    print("Computing: {0}".format(repr(key)))


if __name__ == "__main__":
    # Controls whether to run with/without computing the gramian matrices
    run_gramian = False

    g = da.from_npy_stack("data/pc_rel_10000")
    g = g.persist()
    pcs = compute_pca(g, False)
    pcs = pcs.persist()

    if run_gramian:
        res = pc_relate(pcs, g, maf=0.01)
    else:
        res = pc_relate_no_gramian(pcs, g, maf=0.01)

    # Visualize the taskgraph (run `pip install graphviz` first)
    if run_gramian:
        dask.visualize(res, filename='pc_relate.png', verbose=True)
    else:
        dask.visualize(res, filename='pc_relate_no_gramian.png', verbose=True)

    # Run the computation with profilers
    with Profiler() as prof, ResourceProfiler(dt=0.25) as rprof, CacheProfiler() as cprof, Callback(pretask=printkeys):
        da.compute(res, scheduler="threads")

        # Visualize the profile results
        from dask.diagnostics import visualize
        if run_gramian:
            file_path = "pc_relate_profile.html"
        else:
            file_path = "pc_relate_no_gramian_profile.html"
        visualize([prof, rprof, cprof], file_path=file_path)
