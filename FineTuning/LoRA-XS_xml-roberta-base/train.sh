#!/bin/bash
#SBATCH --gpus=2080ti:1
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/train_%j.log
#SBATCH --account=cil_jobs

module load cuda/12.6.0
source /cluster/courses/cil/envs/etc/profile.d/conda.sh
conda activate ~/lora-env

cd ~/LoRA-XS

python main_glue.py \
  --model_name_or_path xlm-roberta-base \
  --lora_rank 16 \
  --train_file ~/data/train_split.csv \
  --validation_file ~/data/val_split.csv \
  --test_file ~/data/test_inference.csv \
  --do_train --do_eval --do_predict \
  --max_seq_length 256 \
  --per_device_train_batch_size 32 \
  --learning_rate 1e-4 \
  --cls_learning_rate 5e-3 \
  --num_train_epochs 5 \
  --evaluation_strategy epoch \
  --output_dir ./output \
  --ignore_mismatched_sizes \
  --overwrite_output_dir \
  --report_to none \
