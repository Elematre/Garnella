# How to Run Baseline Experiments

This guide explains how to configure your environment and run baseline jobs with `ExperimentBaseline.py`.

## 1. Setup

### 1.1 Configure your environment in `setup.sh`
Update the conda environment name in `setup.sh`:

```bash
conda activate <your-env-name>
```

### 1.2 Create a Hugging Face token
Some models require access to Google Gemma embeddings.

1. Request model access:
   https://huggingface.co/google/embeddinggemma-300m
2. Create a token with read permission:
   https://huggingface.co/settings/tokens
3. Add the token to `setup.sh`:

```bash
export HF_TOKEN=your_token_here
```

## 2. Run a baseline job

From the `Baselines` directory, submit a job with Slurm:

```bash
sbatch -A cil -o ~/repo/Baselines/results/sBatchOut/%j.out ExperimentBaseline.py --model Gemma_LogReg

```
or run from the Terminal

```bash
python ExperimentBaseline.py --model Gemma_MLP_EV

```


## 3. Available baseline models

`ExperimentBaseline.py` currently registers these five models:

```python
BASELINE_REGISTRY = {
    "BoW_LogReg": BoWLogReg,
    "Gemma_LogReg": GemmaLogReg,
    "Gemma_MLP_NoEV": GemmaMLP_NoEV,
    "Gemma_XGBoost": GemmaXGBoost,
    "Gemma_MLP_EV": GemmaMLP_EV,
}
```

These are intentionally kept as a compact set of simple baselines.

## 4. Outputs

- Results CSV: `~/repo/Baselines/results/baseline_results.csv`
- Slurm logs: `~/repo/Baselines/results/sBatchOut/`

The Slurm output logs are useful for debugging failed runs.

## 5. Architecture overview

`ExperimentBaseline.py` runs a model that implements the abstract `BaselineModel` interface.

To add a new baseline:

1. Implement a new model class (for example, similar to `BoWLogReg`).
2. Add it to `BASELINE_REGISTRY` in `ExperimentBaseline.py`.

Each baseline can define its own preprocessing strategy, so model-specific preprocessing is supported.