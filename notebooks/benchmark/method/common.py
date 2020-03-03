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

def filename(basename, ext=None, **props):
    fname = '-'.join([basename] + [f'{k}={v}' for k, v in props.items()])
    return fname + ('' if ext is None else '.' + ext.replace('.', ''))

class Method(object):
    
    def __init__(self, name, relpath):
        self.name = name
        self.relpath = relpath
        
    def module_path(self, module, path):
        return osp.join(ga.BENCHMARK_METHOD_DATA_DIR, module, self.relpath, path)
    
    def module_file(self, module, basename, ext=None, **props):
        return self.module_path(module, filename(basename or '', ext=ext, **props))
    
    def dataset_path(self, basename, ext=None, **props):
        return self.module_file('datasets', basename, ext=ext, **props)
    
    def analysis_path(self, basename, ext=None, **props):
        return self.module_file('analysis', basename, ext=ext, **props)
    
    @property
    def render_dir(self):
        return self.module_path('renders', '')
        
ld_prune_lsh = Method('ld-prune-lsh', 'ld_prune/lsh')
ld_prune_tild = Method('ld-prune-tild', 'ld_prune/tild')

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