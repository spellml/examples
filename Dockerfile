FROM horovod/horovod:sha-825bb8c

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
