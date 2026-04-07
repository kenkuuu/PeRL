#!/bin/bash
# setup_env.sh
# Run inside the Singularity container to create the virtual environment
# and install all packages.
#
# Usage (directly):
#   singularity exec --nv \
#       --bind /data2/kubota/PeRL:/workspace \
#       perl.sif bash /workspace/env/setup_env.sh

set -e

UV=/opt/uv/uv
VENV=/workspace/perl-env
REQUIREMENTS=/workspace/env/requirements_hard.txt

# -------------------------------------------------------------------------
# 0. Sanity checks
# -------------------------------------------------------------------------
if [ ! -x "$UV" ]; then
    echo "[ERROR] uv not found at $UV"
    exit 1
fi

if [ ! -f "$REQUIREMENTS" ]; then
    echo "[ERROR] requirements not found at $REQUIREMENTS"
    exit 1
fi

# -------------------------------------------------------------------------
# 1. Create virtual environment
# -------------------------------------------------------------------------
echo "[1/5] Creating virtual environment at $VENV ..."
$UV venv $VENV --python /usr/bin/python3.11

export VIRTUAL_ENV=$VENV
export PATH=$VENV/bin:$PATH

# -------------------------------------------------------------------------
# 2. Install build dependencies for flash-attn
# -------------------------------------------------------------------------
echo "[2/5] Installing build dependencies ..."
$UV pip install packaging wheel setuptools ninja

# -------------------------------------------------------------------------
# 3. Install PyTorch 2.8.0 (CUDA 12.8)
# -------------------------------------------------------------------------
echo "[3/5] Installing PyTorch 2.8.0 (cu128) ..."
$UV pip install \
    torch==2.8.0 \
    torchvision==0.23.0 \
    torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128

# -------------------------------------------------------------------------
# 4. Install flash-attn
# -------------------------------------------------------------------------
echo "[4/5] Installing flash-attn 2.8.3 ..."
$UV pip install flash-attn==2.8.3 --no-build-isolation

# -------------------------------------------------------------------------
# 5. Install remaining packages from pinned requirements
# -------------------------------------------------------------------------
echo "[5/5] Installing remaining packages from requirements ..."
grep -vE "^(torch|torchvision|torchaudio|flash-attn)==" \
    $REQUIREMENTS \
    > /tmp/requirements_filtered.txt

$UV pip install -r /tmp/requirements_filtered.txt

# -------------------------------------------------------------------------
# Done
# -------------------------------------------------------------------------
echo ""
echo "Setup complete. Verifying installation ..."
python -c "
import torch
import vllm
import flash_attn
print(f'torch      : {torch.__version__}')
print(f'CUDA avail : {torch.cuda.is_available()}')
print(f'vllm       : {vllm.__version__}')
print(f'flash-attn : {flash_attn.__version__}')
"
