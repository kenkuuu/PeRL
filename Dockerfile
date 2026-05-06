FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Tokyo \
    UV_SYSTEM_PYTHON=1

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

# 1. Install PyTorch cu128 first to pin the CUDA version before vLLM pulls its own
RUN uv pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# 2. Install vLLM cu128 wheel; --torch-backend=cu128 tells uv not to replace torch
RUN uv pip install vllm \
    --extra-index-url https://wheels.vllm.ai/cu128/ \
    --torch-backend=cu128

# 3. Reinstall torch cu128 to ensure vLLM did not downgrade it
RUN uv pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128 \
    --reinstall

# 4. Install remaining training dependencies first (trl[vllm] may pull torch variants)
RUN uv pip install \
    accelerate \
    datasets \
    deepspeed \
    fire \
    huggingface-hub \
    math-verify \
    transformers \
    "trl[vllm]" \
    peft \
    wandb

# 5. Reinstall torch cu128 to ensure final torch is cu128 before flash-attn build
RUN uv pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128 \
    --reinstall

# 6. Install flash-attn last so it compiles against the final settled torch
RUN MAX_JOBS=8 uv pip install flash-attn --no-build-isolation

WORKDIR /workspace

CMD ["/bin/bash"]
