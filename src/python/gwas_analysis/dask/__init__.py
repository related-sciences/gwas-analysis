from gwas_analysis.dask import codecs
from numcodecs.registry import register_codec


def get_dask_client(n_workers, processes=True, n_threads=1, max_mem_fraction=.9, plugins=None):
    import psutil
    from dask.distributed import Client

    ml = psutil.virtual_memory().total * max(min(max_mem_fraction, 1), 0)
    ml = str(int(ml // 1e9) // n_workers)
    client = Client(processes=processes, threads_per_worker=n_threads, n_workers=n_workers, memory_limit=ml + 'GB')

    if plugins is not None and 'pgb' in plugins:
        register_codec(codecs.PackGeneticBits)
        client.register_worker_plugin(codecs.CodecPlugin())

    return client