FROM nvidia/cuda:11.4.2-cudnn8-runtime-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    openmpi-bin openmpi-common openssh-client openssh-server libopenmpi-dev \
    cmake \
    python3-pip python3.8-dev python3.8-distutils \
    openssh-server \
    build-essential \
    cmake \
    curl \
    git \
    rsync \
    unzip \
    wget \
    && rm --force --recursive /var/lib/apt/lists/*
RUN python3 -m pip install --upgrade pip
COPY a100-requirements.txt a100-requirements.txt
RUN wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin && \
    mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
    wget -q https://developer.download.nvidia.com/compute/cuda/11.5.0/local_installers/cuda-repo-ubuntu2004-11-5-local_11.5.0-495.29.05-1_amd64.deb && \
    dpkg -i cuda-repo-ubuntu2004-11-5-local_11.5.0-495.29.05-1_amd64.deb && \
    apt-key add /var/cuda-repo-ubuntu2004-11-5-local/7fa2af80.pub && \
    apt-get update && apt-get -y install cuda
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub & \
    add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /" & \
    apt-get update && apt-get -y install libnccl2=2.11.4-1+cuda11.4 libnccl-dev=2.11.4-1+cuda11.4
RUN python3 -m pip install --no-cache-dir \
    -f https://download.pytorch.org/whl/torch_stable.html \
    torch==1.9.1+cu111 \
    torchvision==0.10.1+cu111 \
    torchaudio==0.9.1
RUN update-alternatives --install /usr/bin/python python $(which python3) 20
RUN HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_MPI=1  \
    python3 -m pip install --no-cache-dir horovod==0.23.0
RUN useradd -ms /bin/bash spell
COPY id_rsa /home/spell/.spell/id_rsa
COPY ssh_config /home/spell/.ssh/config
# TODO(aleksey): the SSH test command warns "Failed to add the host to the list of known hosts
# (/home/spell/.ssh/known_hosts)", a permissions error due to the fact that our user spell is
# AFAICT not the owner of the /home/spell/ directory (specifically the /home/spell/.ssh folder
# and its contents). This chown doesn't fix that? Investigate this bug.
#
# TODO(aleksey): it looks like running as a non-root user breaks NVIDIA (nvidia-smi fails?) so
# we'll have to revert this change.
RUN chown -R spell /home/spell/ && chmod 0600 /home/spell/.spell/id_rsa && \
    cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new \
    && echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new \
    && mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config
USER spell
ENV PATH=/home/spell/.local/bin:$PATH
WORKDIR /home/spell/
# TODO(aleksey): current sticking point: Horovod compiles with *neither* NCCL *nor* Torch
# *nor* MPI support. WTF?
# See output of horovodrun --check-build