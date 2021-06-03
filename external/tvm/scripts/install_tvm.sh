#!/bin/bash
set -ex
# https://tvm.apache.org/docs/install/from_source.html#install-from-source
if [[ ! -d "/tmp/tvm" ]]; then
    git clone --recursive https://github.com/apache/tvm /tmp/tvm
fi
apt-get update && \
    apt-get install -y \
        # python3 python3-dev python3-setuptools \
        gcc libtinfo-dev zlib1g-dev \
        build-essential cmake libedit-dev libxml2-dev \
        llvm-6.0 \
        libgomp1  # S0#61786308
if [[ ! -d "/tmp/tvm/build" ]]; then
    mkdir /tmp/tvm/build
fi
cp /tmp/tvm/cmake/config.cmake /tmp/tvm/build
mv /tmp/tvm/build/config.cmake /tmp/tvm/build/~config.cmake && \
    cat /tmp/tvm/build/~config.cmake | \
        sed -E "s|set\(USE_CUDA OFF\)|set\(USE_CUDA ON\)|" | \
        sed -E "s|set\(USE_GRAPH_RUNTIME OFF\)|set\(USE_GRAPH_RUNTIME ON\)|" | \
        sed -E "s|set\(USE_GRAPH_RUNTIME_DEBUG OFF\)|set\(USE_GRAPH_RUNTIME_DEBUG ON\)|" | \
        sed -E "s|set\(USE_LLVM OFF\)|set\(USE_LLVM /usr/bin/llvm-config-6.0\)|" > \
        /tmp/tvm/build/config.cmake
cd /tmp/tvm/build && cmake .. && make -j4
cd /tmp/tvm/python && /opt/conda/envs/spell/bin/python setup.py install --user && cd ..
# NOTE(aleksey): this dependency on pytest is probably accidental, as it isn't documented.
# But without it, the TVM Python package will not import successfully.
/opt/conda/envs/spell/bin/python -m pip install pytest