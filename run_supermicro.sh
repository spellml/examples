#!/bin/bash
set -ex

# # Enable SSH access to the peer blades.
# chmod 0600 id_rsa
# cat << EOT >> /etc/ssh/ssh_config
# Host *
#     User smci
#     AddKeysToAgent yes
#     IdentityFile /spell/examples/id_rsa
# EOT

# Test SSH command to ensure that the peer blades are accessible.
ssh 172.24.118.110 "ip route get 8.8.8.8"

# Test that NCCL is installed and visible to the container
python3 -c "import torch; print(torch.cuda.nccl.version())"

# The OpenMPI that the container finds is old (version 2.something, circa 2017).
# The version on the target machine is new, (version 4.something, circa 2020).
# This causes the verson compat error. For whatever reason, installing the package
# using the apt flag causes this very old version to be installed. Trying to specify
# a version using the --apt flag doesn't work (we don't support version specs for apt
# packages I guess) so we are stuck building our own Docker image instead. Ugh.

# Check the version of OpenMPI that's installed
mpirun --version && ompi_info | grep "Open MPI"

# DEBUG: where is the new openmpi?
find / -path **/openmpi 2>/dev/null

# DEBUG: where it orted?
# find / -path **/bin/orted 2>/dev/null

# DEBUG: do we have two version of mpirun installed?
# which mpicc.openmpi

# # DEBUG: check folder
# ls /usr/include/openmpi
# # DEBUG: check folder 2
# ls /usr/share/openmpi
# # DEBUG: check folder 3
# ls /etc/openmpi

# DEBUG: more folder checks
# ls /usr/lib/x86_64-linux-gnu/openmpi
# ls /usr/lib/x86_64-linux-gnu/openmpi/include/openmpi
# ls /usr/lib/x86_64-linux-gnu/openmpi/lib/openmpi

# DEBUG: is this the orte?
# ls /usr/lib/x86_64-linux-gnu/openmpi/include/openmpi/orte

# DEBUG: check /usr/bin
# ls /usr/bin/
# DEBUG: check
# /usr/bin/mpirun --version

# Horovod instruction
HOROVOD_GPU_OPERATIONS=NCCL \
    horovodrun -np 2 -H localhost:1,172.24.118.110:1 \
    --network-interface enp2s0f1 \
    python3 distributed/pytorch_mnist.py