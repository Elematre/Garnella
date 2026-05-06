# Running LoRA-XS on the ETH Student Cluster

## Overview

We fine-tune XLM-RoBERTa (a multilingual transformer) on the CIL sentiment classification task using LoRA-XS — a parameter-efficient fine-tuning method that only trains 0.22% of the model's parameters (the tiny 16×16 R matrices inserted into attention layers). The other 99.78% of XLM-RoBERTa is completely frozen.

The task is 5-class sentiment classification (labels 0–4, corresponding to 1–5 stars) on product reviews in English and German.

---

## 1. Environment Setup

The course conda environment cannot have packages installed into it, so create your own:

```bash
# Accept conda Terms of Service first (required once)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create environment in home directory
conda create --prefix ~/lora-env python=3.8.13 -y
conda activate ~/lora-env

# Install LoRA-XS requirements
cd ~/LoRA-XS
pip install -r requirements.txt

# Upgrade bitsandbytes — the pinned version is incompatible with the cluster GPU
pip install bitsandbytes --upgrade
```

To activate your environment in future sessions:

```bash
source /cluster/courses/cil/envs/etc/profile.d/conda.sh
conda activate ~/lora-env
```

---

## 2. Data Preparation

Run this once from inside `~/LoRA-XS/` to prepare the data for the model (TODO: update the paths where the new files should get saved at)

```bash
cd ~/LoRA-XS
python prepare_data.py
```

This reads the raw data from the cluster and creates three files in `~/data/`:

- `train_split.csv` — 226,800 labeled reviews for training (90%)
- `val_split.csv` — 25,200 labeled reviews for validation (10%)
- `test_inference.csv` — 168,000 unlabeled reviews for inference (test.csv with dummy label=0 added so the data loader works)

The original `test.csv` is never modified — its `id` column is needed later to build the Kaggle submission.

---

## 3. Storage Management

The cluster has two storage locations:

| Location | Size | Retention |
| --- | --- | --- |
| `/home/<user>` | 20 GB | permanent |
| `/work/scratch/<user>` | 100 GB | auto-deleted nightly |

**Scratch retention policy** — deleted automatically every night at 23:00:

- Under 10 GB → kept for 7 days
- 10–50 GB → kept for 2 days
- Over 50 GB → kept for 1 day

### What goes where

**Conda environment → home** (7 GB )

**HuggingFace model cache → scratch** (2.4 GB, re-downloadable if deleted). Add this to `train.sh` before the python command:

```bash
export HF_HOME=/work/scratch/$USER/huggingface_cache
```

**Output/checkpoints → home** (irreplaceable — don't risk losing trained models to scratch deletion)

### Check storage usage

```bash
quota                                        # overview
du -sh ~/* ~/.cache 2>/dev/null | sort -rh  # breakdown by folder
```

### Clean up after training

Once a run finishes, delete intermediate checkpoints and keep only the final one to save space:

```bash
ls ~/LoRA-XS/output/xlm-roberta-base/None/<run-folder>/
rm -rf <run-folder>/checkpoint-500 <run-folder>/checkpoint-1000  # etc.
```

---

## 4. SLURM Job Script

**Critical settings:**

| Setting | Value | Why |
| --- | --- | --- |
| `--account` | `cil_jobs` | `cil` has 60min limit and kills training mid-run |
| `--gpus` | `2080ti:1` | Default GPU (RTX 5060 Ti, compute capability 12.0) is incompatible with PyTorch 2.2.1 |
| `--partition` | `jobs` | Ensures correct queue |

Create `~/LoRA-XS/train.sh`(you can find the train.sh file in this directory)

Submit:

```bash
sbatch ~/LoRA-XS/train.sh
```

Monitor:

```bash
squeue --me                          # check job status
tail -f ~/LoRA-XS/logs/train_*.log  # watch live output
```

---

## 5. What the Arguments Mean

| Argument | Value | Meaning |
| --- | --- | --- |
| `--model_name_or_path` | `xlm-roberta-base` | Base model — multilingual, handles EN+DE |
| `--lora_rank` | 16 | Size of the trainable R matrix (16×16 = 256 params per layer) |
| `--max_seq_length` | 256 | Reviews truncated to 256 tokens |
| `--per_device_train_batch_size` | 32 | Reviews per gradient update (reduce to 16 if OOM error) |
| `--learning_rate` | 1e-4 | LR for the LoRA-XS R matrices |
| `--cls_learning_rate` | 5e-3 | Higher LR for the 5-class head (randomly initialized) |
| `--num_train_epochs` | 5 | Full passes over the training data |
| `--evaluation_strategy` | epoch | Evaluate on val set after each epoch |
| `--save_steps` | 2000 | Save checkpoint every 2000 steps (less frequent = less disk usage) |
| `--ignore_mismatched_sizes` | — | Required: replaces pretrained 2-class head with 5-class head |
| `--report_to none` | — | Disables wandb (installed version incompatible with new API keys) |

---

## 6. Alternative Models

The task requires a multilingual model because the dataset contains roughly 50% German and 50% English reviews. German-only or English-only models were ruled out for this reason.

The following multilingual models are compatible with `main_glue.py` via the `--model_name_or_path` argument. VRAM estimates are for LoRA-XS training (only the tiny R matrices are trained, so memory usage is much lower than full fine-tuning, but the full model weights still need to be loaded).

| Model | Params | VRAM (LoRA-XS training) | Languages | Notes |
| --- | --- | --- | --- | --- |
| `xlm-roberta-base` | 125M | ~5 GB | 100 | **Used in this setup** |
| `xlm-roberta-large` | 560M | ~10 GB | 100 | Stronger, needs more VRAM — requires `--gpus 2080ti:1` |
| `bert-base-multilingual-cased` | 110M | ~5 GB | 104 | mBERT — weaker than XLM-RoBERTa on most tasks |

To swap models, change the `--model_name_or_path` flag in `train.sh`:

```bash
--model_name_or_path xlm-roberta-large
```

Note: `xlm-roberta-large` fits on the RTX 2080 Ti with batch size 16. Reduce `--per_device_train_batch_size` to 16 if you get an OOM error.

---
## 7. Melis remarks
Hi people! 
nur paar remarks zum set up:
- mir hend uf em cluster nume 20 GB und nach em erste fine tune hani scho 17.5 Gb ufbruuch aber mun eifach die intermediate checkpoints und so no löschen
- Ich bin mer nöd sicher obs i.o isch s'Huggingface model in scratch ztue.. sind 2.4GB die mer spared.. aber ebe d'Sache uf em scratch werded nach enere Wuche glöst.. aber i think it just then gets redownloaded? so should be fine? not sure...
