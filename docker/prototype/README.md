### Genetics API Prototype Images


#### GPU Image

This extends from rapidsai/rapidsai:0.13-cuda10.1-base-ubuntu18.04-py3.7 ([dockerhub tags](https://hub.docker.com/r/rapidsai/rapidsai/tags)).

The system used to test this (Ubuntu 18.04) had NVIDIA driver 430.34 installed (which is only compatible with CUDA Toolkit 10.1).  GPU is single GeForce RTX 2070.


Beyond docker-ce, the only requirement is to have the debian [nvidia-container-runtime](https://nvidia.github.io/nvidia-container-runtime/) package installed on the host system.  With this installed, the nvidia runtime is used automatically now by passing a `--gpus` argument to `docker run`.  See [Access an NVIDIA GPU](https://docs.docker.com/engine/reference/commandline/run/#access-an-nvidia-gpu) in Docker docs for more info.


Build GPU-enabled image:

```
docker build -t gwas-analysis-proto-gpu -f Dockerfile.proto.gpu . 
```

To run:

```
docker run --gpus all --rm -it -p 8890:8888 -p 8687:8787 -p 8686:8786 \
-v /data/disk1/dev:/lab/data \
-v /home/$USER/repos/rs/gwas-analysis:/lab/repos/gwas-analysis \
gwas-analysis-proto-gpu
```
