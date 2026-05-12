#!/bin/bash
#SBATCH --gpus=2080ti:1
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/train_%j.log
#SBATCH --account=cil_jobs

module load cuda/12.6.0
source /cluster/courses/cil/envs/etc/profile.d/conda.sh
conda activate ~/lora-env

export HF_HOME=/work/scratch/$USER/huggingface_cache

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
  --num_train_epochs 15 \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --save_total_limit 3 \
  --load_best_model_at_end True \
  --metric_for_best_model eval_kaggle_score \
  --greater_is_better True \
  --output_dir ./output \
  --ignore_mismatched_sizes \
  --overwrite_output_dir \
  --resume_from_checkpoint "/home/mdietsche/LoRA-XS/output/xlm-roberta-base/None/LoRA_init_svd_rank_16_lr_0.0001_clslr_0.005_seed_42/output_2026-05-03 21:12:32.590610/checkpoint-35000" \
  --report_to none \
