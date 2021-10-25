FROM nvidia/cuda:11.4.2-cudnn8-runtime-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    g++-7 \
    git \
    curl \
    rsync \
    unzip \
    wget \
    openmpi-bin openmpi-common openssh-client openssh-server libopenmpi-dev \
    python3.8 python3.8-dev python3.8-distutils \
    openssh-server \
    cmake \
    && rm --force --recursive /var/lib/apt/lists/*
RUN ln -s /usr/bin/python3.8 /usr/bin/python
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py
# RUN python3 -m pip install --upgrade pip
# RUN update-alternatives --install /usr/bin/python python $(which python3) 20

# Install CUDA Toolkit 11.4
RUN wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin && \
    mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
    wget -q https://developer.download.nvidia.com/compute/cuda/11.4.0/local_installers/cuda-repo-ubuntu2004-11-4-local_11.4.0-470.42.01-1_amd64.deb && \
    dpkg -i cuda-repo-ubuntu2004-11-4-local_11.4.0-470.42.01-1_amd64.deb && \
    apt-key add /var/cuda-repo-ubuntu2004-11-4-local/7fa2af80.pub && \
    apt-get update && \
    apt-get -y install cuda
# Install CUDA Toolkit 11.5; incompatible with out 11.4.2 base, and an 11.5 base Docker image is not out yet, there
# doesn't appear to be an NCCL for this version of CUDA yet either.
# RUN wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin && \
#     mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
#     wget -q https://developer.download.nvidia.com/compute/cuda/11.5.0/local_installers/cuda-repo-ubuntu2004-11-5-local_11.5.0-495.29.05-1_amd64.deb && \
#     dpkg -i cuda-repo-ubuntu2004-11-5-local_11.5.0-495.29.05-1_amd64.deb && \
#     apt-key add /var/cuda-repo-ubuntu2004-11-5-local/7fa2af80.pub && \
#     apt-get update && apt-get -y install cuda

# Install NCCL 2.11.4 for CUDA 11.4
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub & \
    add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /" & \
    apt-get update && apt-get -y install libnccl2=2.11.4-1+cuda11.4 libnccl-dev=2.11.4-1+cuda11.4

# Install PyTorch
RUN python3 -m pip install --no-cache-dir \
    -f https://download.pytorch.org/whl/torch_stable.html \
    torch==1.9.1+cu111 \
    torchvision==0.10.1+cu111 \
    torchaudio==0.9.1

# Install Horovod ("temporarily using CUDA stubs"; this line ripped from https://github.com/horovod/horovod/blob/master/docker/horovod/Dockerfile#L94)
RUN ldconfig /usr/local/cuda/targets/x86_64-linux/lib/stubs && \
    bash -c "HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_PYTORCH=1 pip install --no-cache-dir horovod==0.23.0"
# RUN HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_MPI=1  \
#     python3 -m pip install --no-cache-dir horovod==0.23.0

# Configure SSH
RUN useradd -ms /bin/bash spell
COPY id_rsa /home/spell/.spell/id_rsa
COPY ssh_config /home/spell/.ssh/config
RUN chown -R spell /home/spell/ && chmod 0600 /home/spell/.spell/id_rsa && \
    cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new \
    && echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new \
    && mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

# Switch to Spell user
USER spell
ENV PATH=/home/spell/.local/bin:$PATH
WORKDIR /home/spell/

# Test the framework works correctly.
RUN ldconfig /usr/local/cuda/targets/x86_64-linux/lib/stubs && \
    python -c "import horovod.torch as hvd; hvd.init()" && \
    ldconfig

# TODO(aleksey): current sticking point: Horovod compiles with *neither* NCCL *nor* Torch
# *nor* MPI support. WTF?
# See output of horovodrun --check-build