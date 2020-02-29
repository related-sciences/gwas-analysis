## Docker Setup

Build
```
docker build --build-arg USERNAME=$USER --build-arg USERID=$(id -u) -t gwas-analysis .
```

Run

```
docker run --gpus all --user $(id -u):$(id -g) --rm -ti \
-v /data/disk1/dev:/home/$USER/data \
-v /home/$USER/repos/rs/gwas-analysis:/home/$USER/repos/gwas-analysis \
-v $HOME/.ivy2:/home/$USER/.ivy2 \
-p 8889:8888 -p 4040:4040 -p 4041:4041 -p 8787:8787 -p 8080:8080 -p 8079:8079 \
gwas-analysis
```

TODO: is ```--cap-add SYS_ADMIN``` necessary for async-profiler?  Add if it fails on next run.
