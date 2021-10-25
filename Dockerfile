ARG CUDA_DOCKER_VERSION=11.2.2-devel-ubuntu20.04
FROM nvidia/cuda:${CUDA_DOCKER_VERSION}
ENV DEBIAN_FRONTEND=noninteractive

# Arguments for the build. CUDA_DOCKER_VERSION needs to be repeated because
# the first usage only applies to the FROM tag.
# TensorFlow version is tightly coupled to CUDA and cuDNN so it should be selected carefully
ARG CUDA_DOCKER_VERSION=11.2.2-devel-ubuntu18.04
ARG PYTORCH_VERSION=1.8.1+cu111
ARG TORCHVISION_VERSION=0.9.1+cu111
# NOTE(aleksey): these packages are not even available anymore lol.
# ARG CUDNN_VERSION=8.1.1.33-1+cuda11.2
# ARG NCCL_VERSION=2.8.4-1+cuda11.2

ARG PYTHON_VERSION=3.8

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
    build-essential \
    cmake \
    g++-7 \
    git \
    curl \
    vim \
    wget \
    ca-certificates \
    # libcudnn8=${CUDNN_VERSION} \
    # libnccl2=${NCCL_VERSION} \
    # libnccl-dev=${NCCL_VERSION} \
    libcudnn8 \
    libnccl2 \
    libnccl-dev \
    libjpeg-dev \
    libpng-dev \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-distutils \
    librdmacm1 \
    libibverbs1 \
    libeigen3-dev \
    ibverbs-providers \
    openjdk-8-jdk-headless \
    openssh-client \
    openssh-server \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Open MPI
RUN wget --progress=dot:mega -O /tmp/openmpi-3.0.0-bin.tar.gz https://github.com/horovod/horovod/files/1596799/openmpi-3.0.0-bin.tar.gz && \
    cd /usr/local && \
    tar -zxf /tmp/openmpi-3.0.0-bin.tar.gz && \
    ldconfig && \
    mpirun --version

# Allow OpenSSH to talk to containers without asking for confirmation
RUN mkdir -p /var/run/sshd
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# Install PyTorch
RUN pip install --no-cache-dir \
    torch==${PYTORCH_VERSION} \
    torchvision==${TORCHVISION_VERSION} \
    -f https://download.pytorch.org/whl/${PYTORCH_VERSION/*+/}/torch_stable.html

# Install Horovod, temporarily using CUDA stubs
# WORKDIR /horovod
# COPY horovod .
# RUN cd horovod && ls -a . && python setup.py sdist && \
RUN ldconfig /usr/local/cuda/targets/x86_64-linux/lib/stubs && \
    bash -c "HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_PYTORCH=1 HOROVOD_WITHOUT_GLOO=1 HOROVOD_WITH_MPI=1 HOROVOD_WITHOUT_TENSORFLOW=1 HOROVOD_WITHOUT_MXNET=1 pip install --no-cache-dir horovod" && \
    horovodrun --check-build && \
    ldconfig

# Check the framework is working correctly. Use CUDA stubs to ensure CUDA libs can be found correctly
# when running on CPU machine
RUN ldconfig /usr/local/cuda/targets/x86_64-linux/lib/stubs && \
    python -c "import horovod.torch as hvd; hvd.init()" && \
    ldconfig

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
