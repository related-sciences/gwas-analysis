## Docker Setup

Build
```
docker build --build-arg USERNAME=$USER --build-arg USERID=$(id -u) -t gwas-analysis .
```

Run

```
docker run --user $(id -u):$(id -g) --rm -ti \
-v /data/disk1/dev:/home/$USER/data \
-v /home/$USER/repos/rs/gwas-analysis:/home/$USER/repos/gwas-analysis \
-v $HOME/.ivy2:/home/$USER/.ivy2 \
-p 8889:8888 -p 4040:4040 \
gwas-analysis
```