FROM tensorflow/tensorflow:latest-gpu-py3
MAINTAINER "Spell.run <support@spell.run>"

RUN pip3 install idx2numpy && pip3 install keras
RUN apt-get update && apt-get install wget

WORKDIR /root

COPY workspace .

