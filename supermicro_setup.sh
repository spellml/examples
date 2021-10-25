#!/bin/bash
# Setup scripts. Needs to be run with sudo because it installs many packages.
set -ex

apt-get install -y \
	openmpi-bin openmpi-common openssh-client openssh-server libopenmpi-dev \
	cmake \  # g++ is also needed but seems to already be installed
	python3-pip \  # not installed by default

# Install the CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.5.0/local_installers/cuda-repo-ubuntu2004-11-5-local_11.5.0-495.29.05-1_amd64.deb
dpkg -i cuda-repo-ubuntu2004-11-5-local_11.5.0-495.29.05-1_amd64.deb
apt-key add /var/cuda-repo-ubuntu2004-11-5-local/7fa2af80.pub
apt-get update && apt-get -y install cuda

# Install NCCL
apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
apt-get update && apt install libnccl2=2.11.4-1+cuda11.4 libnccl-dev=2.11.4-1+cuda11.4

# Install Horovod
HOROVOD_GPU_OPERATIONS=NCCL python3 -m pip install horovod
ln -sf /usr/bin/python3 /usr/bin/python  # horovod wants this alternative path

# Add the peer blades to known_hosts so that they can talk to one another.
# NOTE(aleksey): you also need to populated the public key on the worker PMT blades, we do this
# elsewhere.
if [[ ! -d /home/smci/.ssh/ ]]; then
	  mkdir /home/smci/.ssh/
fi
if [[ ! -f /home/smci/.ssh/known_hosts ]]; then
	  touch /home/smci/.ssh/known_hosts
fi
MY_IP=$(ip route get 8.8.8.8 | grep -oP 'src \K[^ ]+')
for BLADE_IP in "172.24.118.175" "172.24.118.181" "172.24.118.110" "172.24.118.156" "172.24.118.101" "172.24.118.208" "172.24.118.152" "172.24.118.212" "172.24.118.151" "172.24.118.168"
do
		if [[ $MY_IP != $BLADE_IP ]]; then
				ssh-keyscan $BLADE_IP >> ~/.ssh/known_hosts
		fi
done

# # disable the password prompt (requires superuser permissions)
# # actually, don't do this!!!
# sed -i 's/PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
