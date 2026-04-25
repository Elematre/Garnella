#!/usr/bin/env python3
"""
Modular experiment runner for 5-class sentiment classification.

This script runs exactly one model per invocation and is designed for cluster usage
(via sbatch or direct CLI). It supports:

1) A gradient-based PyTorch training loop (model: hf-seqclf).
2) Classical sklearn baselines with pluggable text adapters (BoW, TF-IDF, char n-grams).

Core flow:
- Load and preprocess data
- Stratified holdout split (default 90/10)
- Train one selected model
- Evaluate on validation using:
    mae_val = mean_absolute_error(y_val, y_val_pred)
    score_val = 1.0 - (mae_val / 4.0)
    accuracy_val = np.mean(y_val == y_val_pred)
- Optionally run test inference and export predictions
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AdamW,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from repo.utils.preprocessing import preprocess_df


CLASSICAL_MODELS = {
    "logreg",
    "linear-svm",
    "random-forest",
    "mlp",
    "xgboost",
}

TORCH_MODELS = {"hf-seqclf"}

ALL_MODELS = sorted(CLASSICAL_MODELS | TORCH_MODELS)


@dataclass
class ExperimentConfig:
    train_path: str
    test_path: Optional[str]
    output_dir: str
    model: str
    embedding: str
    text_col: str
    label_col: str
    id_col: str
    preprocess_version: int
    test_size: float
    seed: int
    device: str
    num_labels: int
    epochs: int
    batch_size: int
    eval_batch_size: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    max_length: int
    patience: int
    min_delta: float
    num_workers: int
    hf_model_name: str


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="Run one sentiment experiment.")

    parser.add_argument("--train-path", required=True, help="Path to train CSV with labels.")
    parser.add_argument("--test-path", default=None, help="Optional path to test CSV.")
    parser.add_argument("--output-dir", required=True, help="Directory to write artifacts.")

    parser.add_argument(
        "--model",
        required=True,
        choices=ALL_MODELS,
        help="Model to train. Use hf-seqclf for gradient-based PyTorch training.",
    )
    parser.add_argument(
        "--embedding",
        default="bow",
        choices=["bow", "tfidf", "char-ngram"],
        help="Text adapter for classical models (ignored for hf-seqclf).",
    )

    parser.add_argument("--text-col", default="sentence", help="Text column name.")
    parser.add_argument("--label-col", default="label", help="Label column name.")
    parser.add_argument("--id-col", default="id", help="ID column name for exports.")

    parser.add_argument(
        "--preprocess-version",
        type=int,
        default=4,
        help="Preprocessing version from preprocessing.py (1-6).",
    )
    parser.add_argument("--test-size", type=float, default=0.1, help="Validation split fraction.")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")

    parser.add_argument(
        "--device",
        default="auto",
        help="Device for torch path: auto, cpu, cuda, cuda:0, ...",
    )
    parser.add_argument("--num-labels", type=int, default=5, help="Expected number of classes.")

    parser.add_argument("--epochs", type=int, default=4, help="Torch epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Torch train batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=64, help="Torch eval batch size.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Torch learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Torch weight decay.")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Torch warmup ratio.")
    parser.add_argument("--max-length", type=int, default=128, help="Tokenizer max length.")
    parser.add_argument("--patience", type=int, default=2, help="Early stopping patience.")
    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-4,
        help="Minimum score improvement to reset patience.",
    )
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument(
        "--hf-model-name",
        default="bert-base-multilingual-cased",
        help="HF model name for hf-seqclf.",
    )

    args = parser.parse_args()
    return ExperimentConfig(**vars(args))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_flag: str) -> torch.device:
    if device_flag == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_flag)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("Requested CUDA, but it is unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return device


def ensure_dir(path: str) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_json(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    score = float(1.0 - (mae / 4.0))
    accuracy = float(np.mean(y_true_arr == y_pred_arr))

    return {
        "mae": mae,
        "score": score,
        "accuracy": accuracy,
    }


class LabelMapper:
    def __init__(self, labels: Sequence[int], expected_num_labels: int) -> None:
        unique = sorted(pd.Series(labels).dropna().unique().tolist())
        if len(unique) != expected_num_labels:
            print(
                f"Warning: expected {expected_num_labels} classes, found {len(unique)} classes: {unique}"
            )
        self.idx_to_label = unique
        self.label_to_idx = {label: idx for idx, label in enumerate(unique)}

    def encode(self, labels: Sequence[int]) -> np.ndarray:
        return np.asarray([self.label_to_idx[l] for l in labels], dtype=np.int64)

    def decode(self, idxs: Sequence[int]) -> np.ndarray:
        labels = [self.idx_to_label[int(i)] for i in idxs]
        return np.asarray(labels)


class TextAdapter:
    def fit_transform(self, texts: Sequence[str]) -> Any:
        raise NotImplementedError

    def transform(self, texts: Sequence[str]) -> Any:
        raise NotImplementedError


class BowAdapter(TextAdapter):
    def __init__(self) -> None:
        self.vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=10000)

    def fit_transform(self, texts: Sequence[str]):
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts: Sequence[str]):
        return self.vectorizer.transform(texts)


class TfidfAdapter(TextAdapter):
    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(max_features=10000)

    def fit_transform(self, texts: Sequence[str]):
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts: Sequence[str]):
        return self.vectorizer.transform(texts)


class CharNgramAdapter(TextAdapter):
    def __init__(self) -> None:
        self.vectorizer = CountVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=10000)

    def fit_transform(self, texts: Sequence[str]):
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts: Sequence[str]):
        return self.vectorizer.transform(texts)


def make_text_adapter(name: str) -> TextAdapter:
    if name == "bow":
        return BowAdapter()
    if name == "tfidf":
        return TfidfAdapter()
    if name == "char-ngram":
        return CharNgramAdapter()
    raise ValueError(f"Unknown embedding adapter: {name}")


def make_classical_model(name: str):
    if name == "logreg":
        return LogisticRegression(C=1.0, max_iter=1000, n_jobs=-1)
    if name == "linear-svm":
        return LinearSVC(C=1.0, max_iter=2000)
    if name == "random-forest":
        return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=1)
    if name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(128,),
            alpha=1e-2,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            max_iter=300,
            random_state=1,
        )
    if name == "xgboost":
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise RuntimeError(
                "xgboost is not installed, but model=xgboost was requested."
            ) from exc
        return xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            device="cuda",
            n_jobs=-1,
            random_state=1,
        )
    raise ValueError(f"Unknown classical model: {name}")


class EncodedTextDataset(Dataset):
    def __init__(self, encodings: Dict[str, torch.Tensor], labels: Optional[np.ndarray] = None):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {k: v[idx] for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class HFInputAdapter:
    def __init__(self, model_name: str, max_length: int) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def encode(self, texts: Sequence[str]) -> Dict[str, torch.Tensor]:
        return self.tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )


class TorchTrainer:
    def __init__(
        self,
        model: AutoModelForSequenceClassification,
        device: torch.device,
        learning_rate: float,
        weight_decay: float,
        warmup_ratio: float,
        epochs: int,
        patience: int,
        min_delta: float,
    ) -> None:
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.epochs = epochs
        self.patience = patience
        self.min_delta = min_delta

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        val_labels_original: np.ndarray,
        label_mapper: LabelMapper,
    ) -> Tuple[AutoModelForSequenceClassification, Dict[str, float]]:
        self.model.to(self.device)
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        total_steps = max(1, self.epochs * len(train_loader))
        warmup_steps = int(self.warmup_ratio * total_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        best_state = None
        best_metrics: Dict[str, float] = {"score": -np.inf, "mae": np.inf, "accuracy": 0.0}
        no_improve_epochs = 0

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            running_loss = 0.0

            for batch in train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss

                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                running_loss += float(loss.item())

            avg_train_loss = running_loss / max(1, len(train_loader))

            val_pred_idx = self.predict_indices(val_loader)
            val_pred_labels = label_mapper.decode(val_pred_idx)
            val_metrics = compute_metrics(val_labels_original, val_pred_labels)

            print(
                f"Epoch {epoch}/{self.epochs} | "
                f"train_loss={avg_train_loss:.4f} | "
                f"val_score={val_metrics['score']:.4f} | "
                f"val_mae={val_metrics['mae']:.4f} | "
                f"val_acc={val_metrics['accuracy']:.4f}"
            )

            if val_metrics["score"] > best_metrics["score"] + self.min_delta:
                best_metrics = val_metrics
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            if no_improve_epochs >= self.patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self.model, best_metrics

    def predict_indices(self, data_loader: DataLoader) -> np.ndarray:
        self.model.eval()
        preds: List[np.ndarray] = []

        with torch.no_grad():
            for batch in data_loader:
                batch = {k: v.to(self.device) for k, v in batch.items() if k != "labels"}
                logits = self.model(**batch).logits
                preds.append(torch.argmax(logits, dim=-1).cpu().numpy())

        if not preds:
            return np.asarray([], dtype=np.int64)
        return np.concatenate(preds)


def load_dataframe(path: str, required_cols: Sequence[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}")
    return df


def preprocess_text_column(df: pd.DataFrame, text_col: str, version: int) -> pd.Series:
    return preprocess_df(df[text_col].astype(str), version=version)


def run_classical_experiment(
    cfg: ExperimentConfig,
    train_texts: pd.Series,
    val_texts: pd.Series,
    y_train_idx: np.ndarray,
    y_val_original: np.ndarray,
    label_mapper: LabelMapper,
    output_dir: Path,
    test_texts: Optional[pd.Series],
    test_ids: Optional[pd.Series],
) -> None:
    adapter = make_text_adapter(cfg.embedding)
    X_train = adapter.fit_transform(train_texts)
    X_val = adapter.transform(val_texts)

    model = make_classical_model(cfg.model)
    model.fit(X_train, y_train_idx)

    val_pred_idx = model.predict(X_val)
    val_pred = label_mapper.decode(val_pred_idx)
    val_metrics = compute_metrics(y_val_original, val_pred)

    print(
        f"Validation Score: {val_metrics['score']:.4f}, "
        f"MAE: {val_metrics['mae']:.4f}, "
        f"Accuracy: {val_metrics['accuracy']:.4f}"
    )

    metrics_payload = {
        "validation": {
            "score": val_metrics["score"],
            "mae": val_metrics["mae"],
            "accuracy": val_metrics["accuracy"],
        }
    }
    save_json(output_dir / "metrics.json", metrics_payload)

    val_out = pd.DataFrame({"y_true": y_val_original, "y_pred": val_pred})
    val_out.to_csv(output_dir / "val_predictions.csv", index=False)

    model_bundle = {
        "model": model,
        "adapter": adapter,
        "label_mapper": label_mapper,
        "config": asdict(cfg),
    }
    with (output_dir / "model.pkl").open("wb") as f:
        pickle.dump(model_bundle, f)

    if test_texts is not None:
        X_test = adapter.transform(test_texts)
        test_pred_idx = model.predict(X_test)
        test_pred = label_mapper.decode(test_pred_idx)

        out = pd.DataFrame({cfg.id_col: test_ids, cfg.label_col: test_pred})
        out.to_csv(output_dir / "test_predictions.csv", index=False)


def run_torch_experiment(
    cfg: ExperimentConfig,
    train_texts: pd.Series,
    val_texts: pd.Series,
    y_train_idx: np.ndarray,
    y_val_original: np.ndarray,
    label_mapper: LabelMapper,
    output_dir: Path,
    test_texts: Optional[pd.Series],
    test_ids: Optional[pd.Series],
) -> None:
    device = resolve_device(cfg.device)
    print(f"Using device: {device}")

    adapter = HFInputAdapter(cfg.hf_model_name, cfg.max_length)

    train_enc = adapter.encode(train_texts.tolist())
    val_enc = adapter.encode(val_texts.tolist())

    train_ds = EncodedTextDataset(train_enc, labels=y_train_idx)
    val_ds = EncodedTextDataset(val_enc, labels=None)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.hf_model_name,
        num_labels=cfg.num_labels,
    )

    trainer = TorchTrainer(
        model=model,
        device=device,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        epochs=cfg.epochs,
        patience=cfg.patience,
        min_delta=cfg.min_delta,
    )

    model, best_metrics = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        val_labels_original=y_val_original,
        label_mapper=label_mapper,
    )

    print(
        f"Best Validation Score: {best_metrics['score']:.4f}, "
        f"MAE: {best_metrics['mae']:.4f}, "
        f"Accuracy: {best_metrics['accuracy']:.4f}"
    )

    val_pred_idx = trainer.predict_indices(val_loader)
    val_pred = label_mapper.decode(val_pred_idx)

    save_json(output_dir / "metrics.json", {"validation": best_metrics})
    pd.DataFrame({"y_true": y_val_original, "y_pred": val_pred}).to_csv(
        output_dir / "val_predictions.csv", index=False
    )

    torch.save(
        {
            "state_dict": model.state_dict(),
            "label_values": label_mapper.idx_to_label,
            "hf_model_name": cfg.hf_model_name,
            "max_length": cfg.max_length,
        },
        output_dir / "model.pt",
    )

    if test_texts is not None:
        test_enc = adapter.encode(test_texts.tolist())
        test_ds = EncodedTextDataset(test_enc, labels=None)
        test_loader = DataLoader(
            test_ds,
            batch_size=cfg.eval_batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        test_pred_idx = trainer.predict_indices(test_loader)
        test_pred = label_mapper.decode(test_pred_idx)

        out = pd.DataFrame({cfg.id_col: test_ids, cfg.label_col: test_pred})
        out.to_csv(output_dir / "test_predictions.csv", index=False)


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)

    output_dir = ensure_dir(cfg.output_dir)
    timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    required_train_cols = [cfg.text_col, cfg.label_col]
    train_df = load_dataframe(cfg.train_path, required_train_cols)
    test_df: Optional[pd.DataFrame] = None

    if cfg.test_path is not None:
        test_df = load_dataframe(cfg.test_path, [cfg.text_col])

    train_df = train_df.copy()
    train_df["_text"] = preprocess_text_column(train_df, cfg.text_col, cfg.preprocess_version)

    if test_df is not None:
        test_df = test_df.copy()
        test_df["_text"] = preprocess_text_column(test_df, cfg.text_col, cfg.preprocess_version)

    split_train, split_val = train_test_split(
        train_df,
        test_size=cfg.test_size,
        stratify=train_df[cfg.label_col],
        random_state=cfg.seed,
    )

    label_mapper = LabelMapper(split_train[cfg.label_col].tolist(), cfg.num_labels)

    y_train_idx = label_mapper.encode(split_train[cfg.label_col].tolist())
    y_val_original = split_val[cfg.label_col].to_numpy()

    test_texts = test_df["_text"] if test_df is not None else None
    if test_df is not None and cfg.id_col in test_df.columns:
        test_ids = test_df[cfg.id_col]
    elif test_df is not None:
        test_ids = pd.Series(np.arange(len(test_df)), name=cfg.id_col)
    else:
        test_ids = None

    run_meta = {
        "created_at_utc": timestamp,
        "train_rows": int(len(train_df)),
        "split_train_rows": int(len(split_train)),
        "split_val_rows": int(len(split_val)),
        "test_rows": int(len(test_df)) if test_df is not None else 0,
    }
    save_json(output_dir / "config.json", {"config": asdict(cfg), "run_meta": run_meta})

    if cfg.model in CLASSICAL_MODELS:
        run_classical_experiment(
            cfg=cfg,
            train_texts=split_train["_text"],
            val_texts=split_val["_text"],
            y_train_idx=y_train_idx,
            y_val_original=y_val_original,
            label_mapper=label_mapper,
            output_dir=output_dir,
            test_texts=test_texts,
            test_ids=test_ids,
        )
    elif cfg.model in TORCH_MODELS:
        run_torch_experiment(
            cfg=cfg,
            train_texts=split_train["_text"],
            val_texts=split_val["_text"],
            y_train_idx=y_train_idx,
            y_val_original=y_val_original,
            label_mapper=label_mapper,
            output_dir=output_dir,
            test_texts=test_texts,
            test_ids=test_ids,
        )
    else:
        raise ValueError(f"Unsupported model: {cfg.model}")

    print("Done. Artifacts written to:", output_dir)


if __name__ == "__main__":
    main()
