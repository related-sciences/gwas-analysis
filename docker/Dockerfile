FROM debian:stretch
ARG USERNAME
ARG USERID

# Match client user
RUN adduser --disabled-password --gecos '' --home /home/$USERNAME --uid $USERID --shell /bin/bash $USERNAME
RUN usermod -aG sudo $USERNAME
RUN echo "$USERNAME     ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# ----------------------------------------------------------------------------------------------------------------------
# Spark Initialization
# ----------------------------------------------------------------------------------------------------------------------
# from: https://github.com/gettyimages/docker-spark/blob/master/Dockerfile

RUN apt-get update \
 && apt-get install -y locales \
 && dpkg-reconfigure -f noninteractive locales \
 && locale-gen C.UTF-8 \
 && /usr/sbin/update-locale LANG=C.UTF-8 \
 && echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen \
 && locale-gen \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Users with other locales should set this in their derivative image
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

RUN apt-get update \
 && apt-get install -y curl unzip \
    python3 python3-setuptools \
 && ln -s /usr/bin/python3 /usr/bin/python \
 && easy_install3 pip py4j \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# http://blog.stuart.axelbrooke.com/python-3-on-spark-return-of-the-pythonhashseed
ENV PYTHONHASHSEED 0
ENV PYTHONIOENCODING UTF-8
ENV PIP_DISABLE_PIP_VERSION_CHECK 1

# JAVA
RUN apt-get update \
 && apt-get install -y openjdk-8-jre \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# HADOOP
ENV HADOOP_VERSION 3.0.0
ENV HADOOP_HOME /usr/hadoop-$HADOOP_VERSION
ENV HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop
ENV PATH $PATH:$HADOOP_HOME/bin
RUN curl -sL --retry 3 \
  "http://archive.apache.org/dist/hadoop/common/hadoop-$HADOOP_VERSION/hadoop-$HADOOP_VERSION.tar.gz" \
  | gunzip \
  | tar -x -C /usr/ \
 && rm -rf $HADOOP_HOME/share/doc \
 && chown -R root:root $HADOOP_HOME

# SPARK (use Scala 2.11 builds since hail is only built against that)
ENV SPARK_VERSION 2.4.4
ENV SPARK_PACKAGE spark-${SPARK_VERSION}-bin-without-hadoop
ENV SPARK_HOME /usr/spark-${SPARK_VERSION}
ENV SPARK_DIST_CLASSPATH="$HADOOP_HOME/etc/hadoop/*:$HADOOP_HOME/share/hadoop/common/lib/*:$HADOOP_HOME/share/hadoop/common/*:$HADOOP_HOME/share/hadoop/hdfs/*:$HADOOP_HOME/share/hadoop/hdfs/lib/*:$HADOOP_HOME/share/hadoop/hdfs/*:$HADOOP_HOME/share/hadoop/yarn/lib/*:$HADOOP_HOME/share/hadoop/yarn/*:$HADOOP_HOME/share/hadoop/mapreduce/lib/*:$HADOOP_HOME/share/hadoop/mapreduce/*:$HADOOP_HOME/share/hadoop/tools/lib/*"
ENV PATH $PATH:${SPARK_HOME}/bin
RUN curl -sL --retry 3 \
  "https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/${SPARK_PACKAGE}.tgz" \
  | gunzip \
  | tar x -C /usr/ \
 && mv /usr/$SPARK_PACKAGE $SPARK_HOME \
 && chown -R root:root $SPARK_HOME

# Default the driver memory to something very large since the purpose of this container is to run local mode tests
# Note: This is crucial for making hail work by default in local mode
RUN cp $SPARK_HOME/conf/spark-env.sh.template $SPARK_HOME/conf/spark-env.sh && \
  echo "export SPARK_DRIVER_MEMORY=32G" >> $SPARK_HOME/conf/spark-env.sh

# ----------------------------------------------------------------------------------------------------------------------
# Conda Initialization
# ----------------------------------------------------------------------------------------------------------------------
# from https://github.com/ContinuumIO/docker-images/blob/master/miniconda3/debian/Dockerfile

ENV PATH /opt/conda/bin:$PATH
RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion && \
    apt-get clean

# Install depedencides for plotly-orca from https://github.com/plotly/orca/issues/150#issuecomment-519192367
RUN apt-get install xvfb libgconf-2-4 libxss1

# Note that the installation in /opt/conda must be accessible to the login user for environment creation
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.7.12-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy && \
    chown -R $USERNAME:$USERNAME /opt/conda

# ----------------------------------------------------------------------------------------------------------------------
# Image Configuration
# ----------------------------------------------------------------------------------------------------------------------

# SBT
RUN curl -L -o sbt.deb https://dl.bintray.com/sbt/debian/sbt-1.3.3.deb && dpkg -i sbt.deb

# Shell Utilities
RUN apt-get update && apt-get -y install htop vim procps
RUN echo "N" | apt-get install sudo

# PLINK
# Adapted from: https://github.com/GELOG/plink/blob/master/plink-2.0-bin/docker/Dockerfile
# Download is from https://www.cog-genomics.org/plink2 (version = 1.90 beta)
RUN wget http://s3.amazonaws.com/plink1-assets/plink_linux_x86_64_20200107.zip && \
  unzip plink_linux_x86_64_20200107.zip -d /usr/local/plink && \
  rm plink_linux_x86_64_20200107.zip
COPY plink-pipe /usr/local/bin/plink-pipe
RUN chmod a+rx /usr/local/bin/plink-pipe

# R
RUN apt-get -y install r-base

# Ammonite (for Scala 2.11)
# https://github.com/lihaoyi/Ammonite/releases/tag/1.6.7 is last release w/ 2.11 support
RUN echo '#!/usr/bin/env sh' > /usr/local/bin/amm && \
    curl -L https://github.com/lihaoyi/Ammonite/releases/download/1.6.7/2.11-1.6.7 >> /usr/local/bin/amm && \
    chmod +x /usr/local/bin/amm

# Switch to login user for remainder of build
RUN mkdir -p /lab/build
RUN chown $USERNAME:$USERNAME /lab/build
WORKDIR /lab/build
USER $USERNAME

# PLINK env vars
RUN echo '\n\
export PLINK_HOME=/usr/local/plink\n\
export PATH=/usr/local/plink:$PATH\n\
' >> ~/.bashrc
ENV PATH /usr/local/plink:$PATH
ENV PLINK_HOME /usr/local/plink

# Python Environment
COPY environment.yml .
RUN conda env create -f environment.yml
RUN conda init bash
RUN echo "conda activate hail" >> ~/.bashrc
ENV PATH /opt/conda/envs/hail/bin:$PATH

# Jupyter Extensions
RUN conda install -c conda-forge jupyter_contrib_nbextensions && \
  jupyter contrib nbextension install --user && \
  jupyter nbextension enable execute_time/ExecuteTime

# Almond (must install as user to use)
# Install options: https://almond.sh/docs/install-options
RUN SCALA_VERSION=2.11.12 ALMOND_VERSION=0.6.0 && \
    curl -Lo coursier https://git.io/coursier-cli && \
    chmod +x coursier && \
    ./coursier bootstrap \
        -r jitpack \
        -i user -I user:sh.almond:scala-kernel-api_$SCALA_VERSION:$ALMOND_VERSION \
        sh.almond:scala-kernel_$SCALA_VERSION:$ALMOND_VERSION \
        -o almond && \
    ./almond --install --metabrowse true \
    --extra-repository https://jitpack.io
# While somewhat redundant to the coursier arguments above, install a manually defined
# kernel with the same launcher arguments AND JVM arguments, which cannot be specified
# to the install script
COPY kernel-almond.json /home/$USERNAME/.local/share/jupyter/kernels/scala/kernel.json

WORKDIR /home/$USERNAME
RUN mkdir repos data

# Ensure that jupyter defaults to bash shell
ENV SHELL /bin/bash
CMD /bin/bash -c "jupyter lab --allow-root --ip=0.0.0.0"
