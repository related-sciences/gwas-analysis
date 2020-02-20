import os
import os.path as osp
import gwas_analysis.reference_genome as garg

DATASET_HM = 'hapmap'
DATASET_1KG = '1kg'
TEMP_DS_DIR = '/tmp/benchmark_datasets'

HAPMAP_PLINK_FILE = osp.expanduser('~/data/gwas/tutorial/1_QC_GWAS/HapMap_3_r3_1')
DATASET_CONFIG = {
    DATASET_HM: dict(path=HAPMAP_PLINK_FILE, reference_genome=garg.hapmap3_hg18_rg()['name'])
}

def dataset_path(ds_name, ds_sample_rate):
    return osp.join(TEMP_DS_DIR, ds_name + '-' + str(ds_sample_rate))


def get_dask_client(n_workers, processes=True, n_threads=1, max_mem_fraction=.9):
    import psutil
    from numcodecs.registry import register_codec
    from gwas_analysis.dask import codecs
    from dask.distributed import Client
    
    register_codec(codecs.PackGeneticBits)
    ml = psutil.virtual_memory().total * max(min(max_mem_fraction, 1), 0)
    ml = str(int(ml // 1e9) // n_workers)
    client = Client(processes=processes, threads_per_worker=n_threads, n_workers=n_workers, memory_limit=ml + 'GB')
    client.register_worker_plugin(codecs.CodecPlugin())
    return client