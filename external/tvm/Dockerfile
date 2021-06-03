FROM ubuntu:18.04
WORKDIR /spell

# Conda install part
RUN apt-get update && \\
    apt-get install -y wget git && rm -rf /var/lib/apt/lists/*
ENV CONDA_HOME=/root/anaconda/
RUN wget \\
    <https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.3-Linux-x86_64.sh> \\
    && mkdir /root/.conda \\
    && bash Miniconda3-py37_4.8.3-Linux-x86_64.sh -fbp $CONDA_HOME \\
    && rm -f Miniconda3-py37_4.8.3-Linux-x86_64.sh
ENV PATH=/root/anaconda/bin:$PATH
# NOTE: Spell runs will fail if pip3 is not avaiable at the command line.
# conda injects pip onto the path, but not pip3, so we create a symlink.
RUN ln /root/anaconda/bin/pip /root/anaconda/bin/pip3
# TVM install part
COPY environment.yml /tmp/environment.yml
RUN conda env create -n spell -f=/tmp/environment.yml
COPY scripts/install_tvm.sh /tmp/install_tvm.sh
RUN chmod +x /tmp/install_tvm.sh && /tmp/install_tvm.sh