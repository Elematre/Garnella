# Fine-Tuning Results

**Kaggle scoring:** `L = 1 - MAE/4` — easy baseline (grade 4): **0.875**, hard baseline (grade 6): **0.906**

Two different scores are tracked:
- **Val kaggle score** — computed on the *validation set* after each epoch during training (proxy for real performance, available per epoch from updated `main_glue.py`)
- **Kaggle leaderboard score** — computed by Kaggle on the *test set* after submission (one number per submission, the actual grade-relevant score)

> Note: `eval_kaggle_score` is only logged automatically from runs using the updated `main_glue.py`. Run 1 predates this change.

---

## Run 1 — xlm-roberta-base, LoRA-XS, 5 epochs

**Checkpoint:** `checkpoints/xlm-roberta-base_lora-xs_5ep/`

| Setting | Value |
| --- | --- |
| Base model | `xlm-roberta-base` (125M params) |
| Method | LoRA-XS (SVD init, rank 16 → 16×16 R matrices) |
| Trainable params | ~0.22% of model |
| Train / val split | 226,800 / 25,200 (90/10) |
| Batch size | 32 |
| Learning rate (R matrices) | 1e-4 |
| Learning rate (classifier head) | 5e-3 |
| Max seq length | 256 tokens |
| Epochs | 5 |
| GPU | RTX 2080 Ti |
| Training time | ~4 hours |

### Kaggle leaderboard score

| Submission | Score |
| --- | --- |
| `submission_xlm-roberta-base_5ep.csv` | **0.884** |

(Easy baseline: 0.875 ✓ — Hard baseline: 0.906 — gap to close: 0.022)

### Per-epoch validation results

| Epoch | Val Loss | Val Accuracy |
| --- | --- | --- |
| 1 | 1.2704 | 0.4579 |
| 2 | 1.1349 | 0.5191 |
| 3 | 1.0206 | 0.5552 |
| 4 | 0.9493 | 0.5902 | 
| 5 | 0.9081 | **0.6083** |

¹ Val kaggle score not tracked in this run (metric added to `main_glue.py` afterwards).

---

## Run 2 — xlm-roberta-large, LoRA-XS, early stop at epoch 5

**Checkpoint:** `output/xlm-roberta-large/None/LoRA_init_svd_rank_16_lr_0.0001_clslr_0.005_seed_42/`

| Setting | Value |
| --- | --- |
| Base model | `xlm-roberta-large` (560M params) |
| Method | LoRA-XS (SVD init, rank 16 → 16×16 R matrices) |
| Trainable params | ~0.19% of model (24,576 non-classifier params) |
| Train / val split | 226,800 / 25,200 (90/10) |
| Batch size | 16 (effective 32 with gradient_accumulation_steps=2) |
| Learning rate (R matrices) | 1e-4 |
| Learning rate (classifier head) | 5e-3 |
| Max seq length | 256 tokens |
| Epochs | 5 (early stopping, patience=3, max=15) |
| fp16 | True |
| GPU | RTX 2080 Ti |
| Training time | ~7.1 hours |

### Kaggle leaderboard score

| Submission | Score |
| --- | --- |
| *(not yet submitted)* | — |

### Per-epoch validation results

| Epoch | Val Loss | Val Accuracy | Val Kaggle Score | Notes |
| --- | --- | --- | --- | --- |
| 1 | 1.0213 | 0.5970 | **0.8836** | Best checkpoint |
| 2 | — | — | — | |
| 3 | — | — | — | |
| 4 | — | — | — | |
| 5 | 1.1136 | 0.5785 | 0.8802 | Early stop triggered after this epoch |

> Best val kaggle score: **0.8836** (epoch 1) — loaded back as final model via `load_best_model_at_end`
> Epochs 2–4 eval metrics not captured in the log extract; early stopping (patience=3) triggered after epoch 5.

(Easy baseline: 0.875 ✓ — Hard baseline: 0.906 — gap to close: 0.022)
