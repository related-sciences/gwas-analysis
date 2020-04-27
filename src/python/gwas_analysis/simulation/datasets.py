import hail as hl
import numpy as np


def add_default_plink_fields(mt):
    """Add fields to PLINK"""
    return mt.annotate_rows(rsid=hl.null(hl.tstr)).annotate_cols(
        fam_id=hl.null(hl.tstr),
        pat_id=hl.null(hl.tstr),
        mat_id=hl.null(hl.tstr),
        is_female=hl.null(hl.tbool),
        is_case=hl.null(hl.tbool),
    )


def get_ldsim_dataset(n_variants=16, n_samples=4, n_contigs=2, seed=None):
    data = []
    rs = np.random.RandomState(seed)
    for v in range(n_variants):
        for s in range(n_samples):
            for c in range(n_contigs):
                data.append(
                    {
                        "v": f"{c+1}:{v+1}:A:C",
                        "s": f"s{s+1:09d}",
                        "cm": 0.1,
                        "GT": hl.Call([rs.randint(0, 2), rs.randint(0, 2)]),
                    }
                )
    ht = hl.Table.parallelize(
        data, hl.dtype("struct{v: str, s: str, cm: float64, GT: call}")
    )
    ht = ht.transmute(**hl.parse_variant(ht.v))
    mt = ht.to_matrix_table(
        row_key=["locus", "alleles"], col_key=["s"], row_fields=["cm"]
    )
    return add_default_plink_fields(mt)
