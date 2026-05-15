#!/bin/bash
#SBATCH --gpus=5060ti:1
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/slurm/train_%j.log
#SBATCH --account=cil_jobs

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

if [[ -f "$PROJECT_DIR/.my-env" ]]; then
    # shellcheck disable=SC1091
    source "$PROJECT_DIR/.my-env"
fi

FT_USERNAME="${FT_USERNAME:-${USER:-}}"
FT_CONDA_ENV="${FT_CONDA_ENV:-my-cil-env}"
FT_DATA_DIR="${FT_DATA_DIR:-$HOME/data}"
FT_REPORT_TO="${FT_REPORT_TO:-wandb}"

if [[ -z "$FT_USERNAME" ]]; then
    echo "FT_USERNAME is empty. Run ./setUp.sh first."
    exit 1
fi

RUN_ID="$(date +%y%m%d_%H%M%S)"
OUTPUT_ROOT="$PROJECT_DIR/outputs/$FT_USERNAME"
RUN_DIR="$OUTPUT_ROOT/$RUN_ID"

mkdir -p "$PROJECT_DIR/outputs/slurm" "$OUTPUT_ROOT" "$RUN_DIR"


source /etc/profile.d/modules.sh
module load cuda/12.8.1

source /cluster/courses/cil/envs/etc/profile.d/conda.sh
conda activate "$FT_CONDA_ENV"

unset PYTHONPATH
export LD_LIBRARY_PATH="/cluster/data/cuda/12.8.1/lib64:${LD_LIBRARY_PATH:-}"

export HF_HOME="/work/scratch/$FT_USERNAME/huggingface_cache"
export WANDB_DIR="$OUTPUT_ROOT/wandb"
mkdir -p "$HF_HOME" "$WANDB_DIR"

echo "Launching run $RUN_ID for user $FT_USERNAME"
echo "Output directory: $RUN_DIR"

python train.py \
    --model_name_or_path    xlm-roberta-base \
    --train_file            "$FT_DATA_DIR/train_split.csv" \
    --validation_file       "$FT_DATA_DIR/val_split.csv" \
    --test_file             "$FT_DATA_DIR/test_inference.csv" \
    --do_train --do_eval --do_predict \
    --max_seq_length        256 \
    --per_device_train_batch_size 64 \
    --learning_rate         2e-5 \
    --num_train_epochs      1 \
    --evaluation_strategy   epoch \
    --output_dir            "$RUN_DIR" \
    --report_to             "$FT_REPORT_TO" \
    --run_name              "xlmr_full_finetune_${FT_USERNAME}_${RUN_ID}"