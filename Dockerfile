# syntax=docker/dockerfile:1
FROM alchemine/base-cuda:11.8.0-cudnn8-runtime-ubuntu22.04

SHELL ["/bin/bash", "-ic"]
WORKDIR /opt/project

# copy poetry configurations
COPY poetry.lock pyproject.toml /opt/project/

# install python environment
RUN poetry env use python3.9 && \
    poetry install --no-root
