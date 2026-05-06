FROM nvidia/cuda:12.8.1-cudnn9-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Tokyo

# System dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    git \
    git-lfs \
    curl \
    wget \
    build-essential \
    cmake \
    ninja-build \
    libaio-dev \
    openmpi-bin \
    libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /workspace

CMD ["/bin/bash"]
