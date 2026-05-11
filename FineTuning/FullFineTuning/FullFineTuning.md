## 0. activate text-classification environment 

conda activate text-classification

## 1. pre process data

mkdir home/USER/data
change directory in prepare_data.py
run python prepare_data.py

## 2. prepare to run train.sh

login to wandb 

Needed for new GPU: 
pip install 'accelerate>=1.1.0'

## 3. run train.sh
go to Garnella/FineTuning/FullFineTuning and run: 

```bash
sbatch ~/LoRA-XS/train.sh
```


Monitor:

```bash
squeue --me                          # check job status
tail -f /logs/train_*.log  # watch live output
```