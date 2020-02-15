import hail as hl
from typing import *


# Adapted from: https://github.com/macarthur-lab/gnomad_hail/blob/63fd946ab3e46ccd0b7a2109193b42c6d0a19fb9/gnomad_hail/utils/sample_qc.py
def hwe_normalized_pca(
        qc_mt: hl.MatrixTable,
        related_samples_to_drop: Optional[hl.Table] = None,
        n_pcs: int = 10
) -> Tuple[List[float], hl.Table, hl.Table]:
    """
    First runs PCA excluding the given related samples,
    then projects these samples in the PC space to return scores for all samples.
    The `related_samples_to_drop` Table has to be keyed by the sample ID and all samples present in this
    table will be excluded from the PCA.
    The loadings Table returned also contains a `pca_af` annotation which is the allele frequency
    used for PCA. This is useful to project other samples in the PC space.
    :param qc_mt: Input QC MT
    :param related_samples_to_drop: Optional table of related samples to drop
    :param n_pcs: Number of PCs to compute
    :param autosomes_only: Whether to run the analysis on autosomes only
    :return: eigenvalues, scores and loadings
    """
    unrelated_mt = qc_mt

    if related_samples_to_drop:
        unrelated_mt = qc_mt.filter_cols(hl.is_missing(related_samples_to_drop[qc_mt.col_key]))

    pca_evals, pca_scores, pca_loadings = hl.hwe_normalized_pca(unrelated_mt.GT, k=n_pcs, compute_loadings=True)
    pca_af_ht = unrelated_mt.annotate_rows(pca_af=hl.agg.mean(unrelated_mt.GT.n_alt_alleles()) / 2).rows()
    pca_loadings = pca_loadings.annotate(pca_af=pca_af_ht[pca_loadings.key].pca_af)

    if not related_samples_to_drop:
        return pca_evals, pca_scores, pca_loadings
    else:
        related_mt = qc_mt.filter_cols(hl.is_defined(related_samples_to_drop[qc_mt.col_key]))
        related_scores = pc_project(related_mt, pca_loadings)
        pca_scores = pca_scores.union(related_scores)
        return pca_evals, pca_scores, pca_loadings


# Adapted from: https://github.com/macarthur-lab/gnomad_hail/blob/537cb9dd19c4a854a9ec7f29e552129081598399/utils/generic.py#L105
def pc_project(
        mt: hl.MatrixTable,
        loadings_ht: hl.Table,
        loading_location: str = "loadings",
        af_location: str = "pca_af"
) -> hl.Table:
    """
    Projects samples in `mt` on pre-computed PCs.
    :param MatrixTable mt: MT containing the samples to project
    :param Table loadings_ht: HT containing the PCA loadings and allele frequencies used for the PCA
    :param str loading_location: Location of expression for loadings in `loadings_ht`
    :param str af_location: Location of expression for allele frequency in `loadings_ht`
    :return: Table with scores calculated from loadings in column `scores`
    :rtype: Table
    """
    mt = pc_hwe_gt(mt, loadings_ht, loading_location, af_location)
    mt = mt.annotate_cols(scores=hl.agg.array_sum(mt.pca_loadings * mt.GTN))
    return mt.cols().select('scores')


def pc_hwe_gt(
        mt: hl.MatrixTable,
        loadings_ht: hl.Table,
        loading_location: str = "loadings",
        af_location: str = "pca_af"
) -> hl.MatrixTable:
    n_variants = loadings_ht.count()

    mt = mt.annotate_rows(
        pca_loadings=loadings_ht[mt.row_key][loading_location],
        pca_af=loadings_ht[mt.row_key][af_location]
    )

    mt = mt.filter_rows(hl.is_defined(mt.pca_loadings) & hl.is_defined(mt.pca_af) &
                        (mt.pca_af > 0) & (mt.pca_af < 1))

    # Attach normalized entries to be used in projection
    mt = mt.annotate_entries(
        GTN=(mt.GT.n_alt_alleles() - 2 * mt.pca_af) / hl.sqrt(n_variants * 2 * mt.pca_af * (1 - mt.pca_af)))

    return mt
