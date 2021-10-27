#!/bin/bash
set -ex

# NOTE(aleksey): OpenMPI is extremely strongly opinionated about not running MPI processes as a
# root user, even from within a Docker container where this is not a significant danger. Cf.
# https://github.com/open-mpi/ompi/issues/4451.
#
# Historically, the workaround for running as root is to pass an additional flag to the `mpirun`
# command. Horovod does not support this because `horovodrun` does not allow passing such flags
# through to MPI (AFAICT). Recent versions of OpenMPI (v3.0.1 or later) added an alternative path,
# setting two envvars instead. But the version of OpenMPI included in the current `horovod/horovod`
# Dockerfile build is 3.0.0 (which is a full major version behind).
#
# Spell builds its own intermediate container based on the user image which automatically switches
# over to the root user. For now, let's try switching back to the `spell` user we created for this
# purpose in the Dockerfile instead. If this doesn't work, we'll have to switch back to root user
# and figure out how to upgrade OpenMPI (so we have access to the envvars).
#
# You can switch users within a shell script in a persistent manner (TIL), so this necessitates
# using `su - $USER -c '$COMMAND'`.

# Test SSH command to ensure that the peer blades are accessible.
su - spell -c 'ssh 172.24.118.110 "ip route get 8.8.8.8"'

# Test that NCCL is installed and visible to the container
python3.7 -c "import torch; print(torch.cuda.nccl.version())"

# Check the version of OpenMPI that's installed
mpirun --version && ompi_info | grep "Open MPI"

# DEBUG: test local command.
# HOROVOD_GPU_OPERATIONS=NCCL \
#     horovodrun -np 1 -H localhost:1 \
#     python3 distributed/pytorch_mnist.py

# Horovod instruction
su - spell -c 'HOROVOD_GPU_OPERATIONS=NCCL \
    horovodrun -np 2 -H localhost:1,172.24.118.110:1 -p 12345 \
    --network-interface enp2s0f1 \
    python3.7 distributed/pytorch_mnist.py'