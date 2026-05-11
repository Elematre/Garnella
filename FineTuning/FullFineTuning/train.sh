#!/bin/bash
#SBATCH --gpus=5060ti:1
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/train_%j.log
#SBATCH --account=cil_jobs


source /etc/profile.d/modules.sh
module load cuda/12.8.1

source /cluster/courses/cil/envs/etc/profile.d/conda.sh
conda activate /cluster/courses/cil/envs/envs/text-classification


pip install --quiet --user --break-system-packages --force-reinstall --no-deps \
    torch --index-url https://download.pytorch.org/whl/cu128

# point the linker to CUDA 12.8 libs so torch finds the right libcudart
export LD_LIBRARY_PATH=/cluster/data/cuda/12.8.1/lib64:$LD_LIBRARY_PATH
export LD_PRELOAD=/cluster/data/cuda/12.8.1/lib64/libcudart.so:/cluster/data/cuda/12.8.1/lib64/libnvrtc.so:/cluster/data/cuda/12.8.1/lib64/libnvJitLink.so

# make sure Python picks up the cu128 torch before the env's older torch
export PYTHONPATH=/home/$USER/.local/lib/python3.14/site-packages:$PYTHONPATH

# resume WANDB run when running from a checkpoint
# export WANDB_RESUME=must
# export WANDB_RUN_ID=ppdsjl0x

cd ~/Garnella/FineTuning/FullFineTuning

mkdir -p /work/scratch/ehaenni/huggingface_cache
export HF_HOME=/work/scratch/ehaenni/huggingface_cache

python train.py \
    --model_name_or_path    xlm-roberta-base \
    --train_file            ~/data/train_split.csv \
    --validation_file       ~/data/val_split.csv \
    --test_file             ~/data/test_inference.csv \
    --do_train --do_eval --do_predict \
    --max_seq_length        256 \
    --per_device_train_batch_size 64 \
    --learning_rate         2e-5 \
    --num_train_epochs      5 \
    --evaluation_strategy   epoch \
    --output_dir            /work/scratch/ehaenni/output/051126_1527 \
    --report_to wandb \
    --run_name xlmr_full_finetune_051126_1527 \


#cp -r /work/scratch/ehaenni/output/051126_1527/final_model ~/Garnella/FineTuning/FullFineTuning/051126_1527/final_model
#cp -r /work/scratch/ehaenni/output/051126_1527/checkpoint-17720 ~/Garnella/FineTuning/FullFineTuning/051126_1527/checkpoint-17720


# cp -r /work/scratch/ehaenni/output/051126_1527/predictions.csv ~/Garnella/FineTuning/FullFineTuning/051126_1527/predictions.csv