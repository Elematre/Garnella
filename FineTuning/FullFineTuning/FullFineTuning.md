# Full Fine-Tuning Quickstart


## 1. Prepare Data

```bash
mkdir -p ~/data
cd ~/Garnella/FineTuning/LoRA-XS
python prepare_data.py
```

## 2. Run SetUp.sh
```bash
chmod +x setUp.sh train.sh
./setUp.sh
```
This extracts your username, wandb etc, creates a conda environment.


## 3. Run
```bash
sbatch train.sh
```

```bash
squeue --me      # check job status
scancel <job_id> # cancel a job
```

---

## Troubleshooting

### PyTorch / CUDA version mismatch

If your job fails instantly with an error like: AssertionError: Expected CUDA 12.8 for RTX 5060 Ti, got 12.6 — check 'module load cuda/12.8.1'

Reinstall PyTorch with the correct CUDA build:

```bash
conda activate my-cil-env
pip install torch \
    --index-url https://download.pytorch.org/whl/cu128 \
    --force-reinstall --no-user
```