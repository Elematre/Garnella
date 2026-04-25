

from __future__ import annotations
import os
import gc
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)

NUM_LABELS = 5


# ---------------------------------------------------------------
# Shared helpers — consider moving to utils.py later
# ---------------------------------------------------------------

def build_text(df: pd.DataFrame, title_col="title", body_col="text",
               sep=" </s> ") -> pd.Series:
    """Concatenate title + body with a separator. Robust to missing title col."""
    if title_col and title_col in df.columns:
        title = df[title_col].fillna("").astype(str)
        body = df[body_col].fillna("").astype(str)
        return (title + sep + body).str.strip()
    return df[body_col].fillna("").astype(str)


def expected_value_decode(logits, num_labels: int = NUM_LABELS) -> np.ndarray:
    """MAE-aware decoding: E[y] = Σ i·p_i, then round. Usually beats argmax."""
    logits_t = torch.as_tensor(logits, dtype=torch.float32)
    probs = torch.softmax(logits_t, dim=-1).numpy()
    expected = (probs * np.arange(num_labels)).sum(axis=-1)
    return np.clip(np.round(expected), 0, num_labels - 1).astype(int)


def _score(preds, labels) -> float:
    """Competition metric: 1 - MAE / 4."""
    return 1 - np.mean(np.abs(np.asarray(preds) - np.asarray(labels))) / 4


def _metrics_fn(num_labels: int = NUM_LABELS):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds_arg = np.argmax(logits, axis=-1)
        preds_exp = expected_value_decode(logits, num_labels)
        return {
            "score_argmax": _score(preds_arg, labels),
            "score_expected": _score(preds_exp, labels),
        }
    return compute_metrics


# ---------------------------------------------------------------
# Core fine-tune function
# ---------------------------------------------------------------

def finetune(
    model_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame | None = None,
    *,
    tag: str | None = None,
    label_col: str = "label",
    title_col: str | None = "title",
    body_col: str = "text",
    max_length: int = 256,
    batch_size: int = 32,
    grad_accum: int = 1,
    eval_batch: int = 64,
    learning_rate: float = 2e-5,
    num_epochs: int = 3,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    output_dir: str = "./checkpoints",
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Fine-tune a HuggingFace encoder for 5-way sentiment classification.

    Returns a dict with the same `validation_score` key as your existing
    train_loop results, plus logits (useful for ensembling later).
    """
    name = tag or model_name.split("/")[-1]
    run_dir = os.path.join(output_dir, name)

    # Reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Build text column (don't mutate caller's dfs)
    train_df = train_df.assign(_text=build_text(train_df, title_col, body_col))
    val_df = val_df.assign(_text=build_text(val_df, title_col, body_col))
    if test_df is not None:
        test_df = test_df.assign(_text=build_text(test_df, title_col, body_col))

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def to_hf(df: pd.DataFrame, with_labels: bool = True) -> Dataset:
        data = {"text": df["_text"].tolist()}
        if with_labels:
            data["labels"] = df[label_col].astype(int).tolist()
        return Dataset.from_dict(data)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    train_ds = to_hf(train_df).map(tok, batched=True, remove_columns=["text"])
    val_ds = to_hf(val_df).map(tok, batched=True, remove_columns=["text"])
    test_ds = None
    if test_df is not None:
        test_ds = to_hf(test_df, with_labels=False).map(
            tok, batched=True, remove_columns=["text"]
        )

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=NUM_LABELS
    )

    # Training args
    args = TrainingArguments(
        output_dir=run_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="score_expected",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        seed=seed,
        report_to="none",
        disable_tqdm=not verbose,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=_metrics_fn(),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )

    if verbose:
        print(f"  [{name}] fine-tuning...")
    trainer.train()

    # Validation
    val_out = trainer.predict(val_ds)
    val_logits = val_out.predictions
    val_labels = val_df[label_col].astype(int).values
    val_preds_exp = expected_value_decode(val_logits)
    val_preds_arg = np.argmax(val_logits, axis=-1)
    val_score = _score(val_preds_exp, val_labels)
    val_score_arg = _score(val_preds_arg, val_labels)

    # Test
    test_logits = None
    test_preds = None
    if test_ds is not None:
        test_logits = trainer.predict(test_ds).predictions
        test_preds = expected_value_decode(test_logits)

    if verbose:
        print(f"  [{name}] val: {val_score:.4f} (expected) | "
              f"{val_score_arg:.4f} (argmax)")

    # Free GPU memory before the next run
    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model_name": name,
        "validation_score": val_score,           # matches your existing schema
        "validation_score_argmax": val_score_arg,
        "val_predictions": val_preds_exp,
        "val_logits": val_logits,
        "test_predictions": test_preds,
        "test_logits": test_logits,
    }


# ---------------------------------------------------------------
# Loop — parallel to your train_loop
# ---------------------------------------------------------------

def finetune_loop(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    configs: list[dict],
    verbose: bool = True,
) -> list[dict]:
    """Run a list of fine-tuning configs, return list of result dicts.

    Each config is a kwargs dict for finetune(); 'model_name' is required.
    Pass 'tag' to disambiguate multiple runs of the same base model.
    """
    results = []
    for i, cfg in enumerate(configs, 1):
        label = cfg.get("tag") or cfg["model_name"]
        if verbose:
            print(f"\n[{i}/{len(configs)}] {label}")
        result = finetune(
            train_df=train_df, val_df=val_df, test_df=test_df,
            verbose=verbose, **cfg,
        )
        results.append(result)
    return results