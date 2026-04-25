#!/bin/bash
# setup.sh — Run once to configure your environment for the GPU cluster

set -e  # Exit on error

echo "=== Setting up environment ==="

# 1. Create conda env from file
conda env create -f environment.yml
conda activate <your-env-name>

# 2. Reinstall torch with CUDA 12.8 support (cluster-specific hack)
python -m pip install --user --break-system-packages --force-reinstall --no-deps \
    torch --index-url https://download.pytorch.org/whl/cu128

# 3. Append cluster env vars to ~/.bashrc (idempotent — won't duplicate)
MARKER="# === CIL Cluster Setup ==="
if ! grep -qF "$MARKER" ~/.bashrc; then
    cat >> ~/.bashrc << 'EOF'

# === CIL Cluster Setup ===
export CUDA_HOME=/cluster/data/cuda/12.8.1
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export LD_PRELOAD=$CUDA_HOME/lib64/libcudart.so:$CUDA_HOME/lib64/libnvrtc.so:$CUDA_HOME/lib64/libnvJitLink.so
export PYTHONPATH=$HOME/.local/lib/python3.14/site-packages:${PYTHONPATH:-}
export HF_TOKEN=your_token_here
EOF
    echo "Added env vars to ~/.bashrc — run: source ~/.bashrc"
else
    echo "Env vars already in ~/.bashrc, skipping."
fi
