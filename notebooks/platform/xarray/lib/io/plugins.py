from .codecs import PackGeneticBits


def create_dask_codec_plugin():

    from dask.distributed import WorkerPlugin
    class DaskCodecPlugin(WorkerPlugin):

        def setup(self, worker):
            from numcodecs.registry import register_codec
            register_codec(PackGeneticBits)
    return DaskCodecPlugin()