#!/bin/bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MY_ENV_FILE="$PROJECT_DIR/.my-env"
FT_ENV_FILE="$PROJECT_DIR/FTEnv.yml"

cd "$PROJECT_DIR"

DEFAULT_USERNAME="${USER:-}"
read -r -p "Cluster username [${DEFAULT_USERNAME}]: " INPUT_USERNAME
FT_USERNAME="${INPUT_USERNAME:-$DEFAULT_USERNAME}"

if [[ -z "$FT_USERNAME" ]]; then
    echo "Username is required."
    exit 1
fi

if [[ ! -f "$FT_ENV_FILE" ]]; then
    echo "ERROR: $FT_ENV_FILE not found"
    exit 1
fi

FT_CONDA_ENV="my-cil-env"
FT_DATA_DIR="$HOME/data"

mkdir -p "$PROJECT_DIR/outputs/slurm" "$PROJECT_DIR/outputs/$FT_USERNAME"

cat > "$MY_ENV_FILE" <<EOF
export FT_USERNAME="$FT_USERNAME"
export FT_CONDA_ENV="$FT_CONDA_ENV"
export FT_DATA_DIR="$FT_DATA_DIR"
export FT_REPORT_TO="wandb"
EOF

echo "Wrote configuration to $MY_ENV_FILE"

source /cluster/courses/cil/envs/etc/profile.d/conda.sh

# =====================================================================
# ENVIRONMENT HANDLING BLOCK (UPDATED)
# =====================================================================
if conda env list | grep -q "^$FT_CONDA_ENV "; then
    echo "--------------------------------------------------------"
    echo "⚠️ Environment '$FT_CONDA_ENV' already exists."
    echo "Skipping update. Please make sure your local environment"
    echo "is correct and matches the shared configuration."
    echo "--------------------------------------------------------"
else
    echo "Creating new environment from $FT_ENV_FILE..."
    conda env create -f "$FT_ENV_FILE" -n "$FT_CONDA_ENV" --yes
fi

echo "Activating '$FT_CONDA_ENV' to run integrity checks..."

# Safely source conda functionality inside the script
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "$FT_CONDA_ENV"

# Cleanly install accelerate into the activated environment (Fast if already satisfied)
echo "Ensuring accelerate>=1.1.0 is installed..."
pip install --no-user "accelerate>=1.1.0"

# Push the script into a safe directory (like the user home folder) where no 'torch' folders exist
pushd ~ > /dev/null

echo "Validating PyTorch inside environment..."
python -c "
import sys
try:
    import torch
    print(f'✅ PyTorch successfully loaded! (v{torch.__version__})')
    
    if torch.cuda.is_available():
        print(f'✅ CUDA is available!')
        print(f'   Device Name: {torch.cuda.get_device_name(0)}')
        print(f'   Device Count: {torch.cuda.device_count()}')
    else:
        print('❌ WARNING: CUDA is NOT available to PyTorch.')
        print('   Your code will fall back to CPU. Check your cluster CUDA modules.')
        
except ImportError:
    print('❌ ERROR: PyTorch is not installed in this environment.')
    sys.exit(1)
except Exception as e:
    print(f'❌ ERROR: PyTorch loaded but threw an exception:\n{e}')
    sys.exit(1)
"
# Pull the script back to your repository folder to finish up
popd > /dev/null

read -r -p "Run 'wandb login' now? [y/N]: " LOGIN_WANDB
if [[ "$LOGIN_WANDB" =~ ^[Yy]$ ]]; then
    wandb login
fi

read -r -p "Run Hugging Face login now? [y/N]: " LOGIN_HF
if [[ "$LOGIN_HF" =~ ^[Yy]$ ]]; then
    if command -v hf >/dev/null 2>&1; then
        hf auth login
    elif command -v huggingface-cli >/dev/null 2>&1; then
        huggingface-cli login
    else
        echo "No Hugging Face CLI found. Install with: pip install huggingface_hub"
    fi
fi

echo
echo "Setup complete."
echo "Training output and SLURM logs will be under: $PROJECT_DIR/outputs/$FT_USERNAME"
echo "Start training with: sbatch train.sh"
echo "Monitor jobs with: squeue --me"
echo "Monitor logs with: tail -f outputs/slurm/train_*.log"