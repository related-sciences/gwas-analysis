## Performance Testing Notes


### async-profiler


[async-profiler](https://github.com/jvm-profiling-tools/async-profiler) is not currently installed as part of the Docker build, but it can be setup as follows:

```
#Install this as root
root> apt-get install openjdk-8-dbg

# Install async-profiler as user
user>
mkdir -p ~/apps/async-profiler; cd ~/apps/async-profiler
wget https://github.com/jvm-profiling-tools/async-profiler/releases/download/v1.2/async-profiler-1.2-linux-x64.zip
unzip async-profiler-1.2-linux-x64.zip
```

To profile an application, run both the profiler and the app as root (it will not work otherwise):

```
sudo su
java -Xmx64G -jar $(which amm) repos/gwas-analysis/notebooks/tutorial/ext/snpseek/plink-file-reader.sc
# amm repos/gwas-analysis/notebooks/tutorial/ext/snpseek/plink-file-reader.sc  # Gives OOM

# In another tab:
cd /home/eczech/apps/async-profiler
./profiler.sh -d 60 -f /home/eczech/data/tmp/flamegraph.svg
```