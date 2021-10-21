#!/bin/bash
set -ex

# Enable SSH access to the peer blades.
chmod 0600 id_rsa
cat << EOT >> /etc/ssh/ssh_config
Host *
    User smci
    AddKeysToAgent yes
    IdentityFile /spell/examples/id_rsa
EOT

# Test SSH command to ensure that the peer blades are accessible.
ssh 172.24.118.110 "ip route get 8.8.8.8"

# Test that NCCL is installed and visible to the container
python -c "import torch; print(torch.cuda.nccl.version())"

# Horovod instruction
HOROVOD_GPU_OPERATIONS=NCCL \
    horovodrun -np 2 -H localhost:1,172.24.118.110:1 \
    python distributed/pytorch_mnist.py
