import os
import os.path as osp
import gwas_analysis as ga
import gwas_analysis.nb as ganb
import gwas_analysis.reference_genome as garg

DATASET_HM = 'hapmap'
DATASET_1KG = '1kg'
DATASET_SIM = 'sim'
HAPMAP_PLINK_FILE = osp.join(ga.TUTORIAL_1_QC_DIR, 'HapMap_3_r3_1')

DATASET_CONFIG = {
    DATASET_HM: dict(path=HAPMAP_PLINK_FILE, reference_genome=garg.hapmap3_hg18_rg()['name']),
    DATASET_SIM: dict(path=None, reference_genome='GRCh38')
}

def filename(filename, **props):
    return '-'.join([filename] + [f'{k}={v}' for k, v in props.items()])

def module_path(module, fname, ext=None, **props):
    # Example: $REPO/notebooks/benchmark/method/ld_prune/lsh/04-analysis.ipynb -> ld_prune/lsh
    relpath = osp.dirname(osp.relpath(ganb.get_notebook_path(), ga.BENCHMARK_METHOD_DIR))
    fname = filename(fname, **props) + ('' if ext is None else '.' + ext.replace('.', ''))
    # Example: $DATA_DIR/gwas/benchmark/$module/ld_prune/lsh/$filename
    return osp.join(ga.BENCHMARK_METHOD_DATA_DIR, module, relpath, fname)

def dataset_path(ds_name, **props):
    return module_path('datasets', ds_name, **props)


def hash_collision_probability(a, h, g):
    """ Compute probability of hash collision for signed random projects (aka SimHash)
    
    See: Similarity Estimation Techniques from Rounding Algorithms (Charikar 2002)
    Args:
        a: Cosine similarity (pearson correlation) between vectors/data points
        h: Number of hash bits
        g: Number of composite hashes (w/ OR amplification)
    """
    import numpy as np
    return 1 - (1 - (1 - np.arccos(a) / np.pi)**h)**g