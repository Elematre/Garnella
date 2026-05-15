"""
train.py
--------
Full fine-tuning of XLM-RoBERTa-base for 5-class sentiment classification.
Arguments mirror the LoRA-XS main_glue.py interface so both experiments
are launched the same way and results are directly comparable.
Expected CSV schema (train / val / test):  sentence, label
Test labels are dummies (0); predictions are written to output_dir/predictions.csv
which can be joined with the original test.csv on row order to build a Kaggle
submission.
"""
import argparse
import logging
import os
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset
from transformers import (
AutoTokenizer,
AutoModelForSequenceClassification,
TrainingArguments,
Trainer,
EarlyStoppingCallback,
DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, mean_absolute_error

NUM_LABELS = 5


# ── Args ──────────────────────────────────────────────────────────────────────
# define what command-line flags the script accepts
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path",         default="xlm-roberta-base")
    # data files
    p.add_argument("--train_file",                 required=True)
    p.add_argument("--validation_file",            required=True)
    p.add_argument("--test_file",                  required=True)
    # what mode to run in (train / eval / predict)
    # store_true" = flag is a boolean switch (can just write --do_train -> sets it to True)
    p.add_argument("--do_train",                   action="store_true")
    p.add_argument("--do_eval",                    action="store_true")
    p.add_argument("--do_predict",                 action="store_true")
    # max num of tokens per sentence
    # 128 is a practical choice for a 2080 Ti (11GB VRAM) -> could go to 256 but uses more memory
    p.add_argument("--max_seq_length",             type=int,   default=128)
    p.add_argument("--per_device_train_batch_size",type=int,   default=32)
    p.add_argument("--learning_rate",              type=float, default=2e-5)
    p.add_argument("--num_train_epochs",           type=int,   default=5)
    p.add_argument("--evaluation_strategy",        type=str,   default="epoch")
    p.add_argument("--output_dir",                 default="./output")
    p.add_argument("--report_to",                  default="none")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--seed",                       type=int,   default=42)
    return p.parse_args()

# ── Dataset ───────────────────────────────────────────────────────────────────
# PyTorch requires data to be wrapped in a Dataset class with __len__ and __getitem__ methods
class ReviewDataset(Dataset):
    # convert all sentences to token IDs once, before training
    def __init__(self, df, tokenizer, max_length):
        self.encodings = tokenizer(
            df["sentence"].astype(str).tolist(),
            truncation=True,
            padding=False,
            max_length=max_length,)
        self.labels = df["label"].astype(int).tolist()
    
    def __len__(self):
        return len(self.labels)
    # self.encodings contains multiple arrays: input_ids, attention_mask, token_type_ids. 
    # Need to return a dict of tensors for each item, plus the label tensor
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# ── Metrics ───────────────────────────────────────────────────────────────────
# use our MAE score
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    mae   = mean_absolute_error(labels, preds)
    score = 1.0 - (mae / 4.0) 
    return {
    "score":    score,
    "mae":      mae,
    "accuracy": accuracy_score(labels, preds),
        }
# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # configure logging so logger.info() actually writes to the log file
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger(__name__)

    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    ### sanity checks 
    assert torch.cuda.is_available(), \
    "CUDA not available"
    assert torch.version.cuda.startswith("12.8"), \
    f"Expected CUDA 12.8 for RTX 5060 Ti, got {torch.version.cuda} — check 'module load cuda/12.8.1'"
    assert "5060" in torch.cuda.get_device_name(0), \
    f"Expected RTX 5060 Ti, got {torch.cuda.get_device_name(0)} — wrong GPU allocated"
    logger.info(f"PyTorch version : {torch.__version__}")
    logger.info(f"CUDA version    : {torch.version.cuda}")
    logger.info(f"GPU             : {torch.cuda.get_device_name(0)}")


    logger.info("Loading tokenizer and model")
    # download weights from HuggingFace Hub or load from a local path
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    # takes the base model (XLM-RoBERTa) and replaces its output head with a new linear layer that outputs 5 scores instead.
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=NUM_LABELS,
        ignore_mismatched_sizes=True,
    )

    logger.info("Loading datasets")
    train_df = pd.read_csv(args.train_file)
    val_df   = pd.read_csv(args.validation_file)
    test_df  = pd.read_csv(args.test_file)
    train_dataset = ReviewDataset(train_df, tokenizer, args.max_seq_length)
    val_dataset   = ReviewDataset(val_df,   tokenizer, args.max_seq_length)
    test_dataset  = ReviewDataset(test_df,  tokenizer, args.max_seq_length)
    
    # here lies most finetuning decicions!!
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        # use twice as many eval batch size to speed up evaluation (no backprop, so less memory needed)
        per_device_eval_batch_size=args.per_device_train_batch_size * 2,
        # how big each weight update is. 
        # For full fine-tuning of a pretrained transformer, 1e-5 to 3e-5 is the standard safe range
        learning_rate=args.learning_rate,
        # L2 regularization, which slightly penalizes large weights and acts as a mild overfitting guard. 
        # 0.01 is a standard default for transformer fine-tuning
        weight_decay=0.01,
        # for first 5% of training steps, learning rate ramps up from 0 to learning_rate. 
        # -> protects pretrained weights at the very start when randomly initialized classification head is producing chaotic gradients that could destabilize the whole model
        warmup_steps=0.05,
        # after warmup, the LR decays following a cosine curve down to ~0
        # alternative is "linear" which decays in a straight line. 
        # -> Cosine tends to perform slightly better in practice because it stays near the peak LR longer before decaying sharply at the end.
        lr_scheduler_type="cosine",
        fp16=torch.cuda.is_available(),
        # evaluate and save a checkpoint after every "evaluation_strategy" (epoch)
        eval_strategy=args.evaluation_strategy,
        save_strategy=args.evaluation_strategy,
        # automatically restore the checkpoint with the best score in the end
        load_best_model_at_end=True,
        # optimize for Kaggle competion score (1-MAE)
        metric_for_best_model="score",
        # higher score is better
        greater_is_better=True,
        logging_steps=100,
        # only keep the 4 most recent checkpoints on disk
        save_total_limit=4,
        seed=args.seed,
        # reports to wandb
        report_to=args.report_to,
        run_name=args.run_name,
        # uses 2 CPU threads to load and tokenize batches in parallel while the GPU is training.
        # -> was suggested by system
        dataloader_num_workers=2,
        )
    # HuggingFace wrapper that runs the full training loop
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        # takes a batch of variable-length tokenized sequences and pads them all to the same length (the longest in that batch)
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        # monitors score on  validation set after each epoch
        # If it doesn't improve for 2 epochs in a row, training stops
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )
    
    # In train.sh all three are passed, so all three run in sequence
    # ── Train ─────────────────────────────────────────────────────────────────
    if args.do_train:
        logger.info("Starting training")
        trainer.train()
        trainer.save_model(os.path.join(args.output_dir, "final_model"))
        tokenizer.save_pretrained(os.path.join(args.output_dir, "final_model"))
    # ── Eval ──────────────────────────────────────────────────────────────────
    if args.do_eval:
        logger.info("Evaluating on validation set")
        metrics = trainer.evaluate()
        logger.info(metrics)
    # ── Predict ───────────────────────────────────────────────────────────────
    if args.do_predict:
        logger.info("Running prediction on test set")
        predictions = trainer.predict(test_dataset)
        preds = np.argmax(predictions.predictions, axis=-1)

        submission = pd.DataFrame({
            "id": test_df.index,
            "label": preds
        })

        out_path = Path(args.output_dir) / "predictions.csv"
        submission.to_csv(out_path, index=False)
        logger.info(f"Predictions saved to {out_path}")

if __name__ == "__main__":
    main()