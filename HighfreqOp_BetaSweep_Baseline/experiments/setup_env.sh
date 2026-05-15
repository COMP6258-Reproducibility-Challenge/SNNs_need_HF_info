#!/bin/bash
# =============================================================================
# setup_env.sh  —  One-time environment setup for Iridis (Southampton HPC)
#
# Run this ONCE from your home directory BEFORE submitting any SLURM jobs:
#   bash setup_env.sh
# =============================================================================

set -e

# ── 1. Load modules ────────────────────────────────────────────────────────────
module purge
module load conda/python3      # confirmed name on Iridis loginX001

# ── 2. Find CUDA module (check and load whichever exists) ─────────────────────
# Run: module avail cuda   to see options. Load the one that exists:
for cuda_mod in cuda/12.1 cuda/11.8 cuda/12.0 cuda/12.2 cuda/11.7; do
    if module load "$cuda_mod" 2>/dev/null; then
        echo "[setup] Loaded CUDA module: $cuda_mod"
        break
    fi
done

# ── 3. Create conda environment ────────────────────────────────────────────────
ENV_NAME="maxformer"

# Ensure channels are configured (fixes NoChannelsConfiguredError on fresh installs)
conda config --add channels defaults 2>/dev/null || true
conda config --add channels conda-forge 2>/dev/null || true

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[setup] Conda env '${ENV_NAME}' already exists. Skipping creation."
else
    echo "[setup] Creating conda env '${ENV_NAME}'..."
    conda create -y -n ${ENV_NAME} python=3.9
fi

# ── 4. Activate and install packages ──────────────────────────────────────────
source activate ${ENV_NAME}

echo "[setup] Installing PyTorch (CUDA 12.1)..."
pip install torch==2.1.0 torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cu121 \
    --quiet

echo "[setup] Installing MaxFormer dependencies..."
pip install \
    timm==0.6.12 \
    spikingjelly==0.0.0.0.12 \
    "opencv-python-headless==4.8.1.78" \
    einops \
    PyYAML \
    Pillow \
    six \
    --quiet

echo "[setup] Installing analysis/plotting dependencies..."
pip install \
    matplotlib \
    seaborn \
    pandas \
    scipy \
    numpy \
    --quiet

pip install wandb --quiet
export WANDB_MODE=disabled

echo ""
echo "[setup] Done! Environment '${ENV_NAME}' is ready."
echo ""
echo "[setup] Test your install with:"
echo "  source activate maxformer"
echo "  python -c \"import torch; import timm; import spikingjelly; print('OK')\""
