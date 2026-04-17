import numpy as np
import torch
import os
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.trainer import SentenceTransformerTrainer
from datasets import Dataset

CACHE_DIR = "./embedding_cache"


def _save_cache(name, train_emb, val_emb):
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(os.path.join(CACHE_DIR, f"{name}_train.npy"), train_emb)
    np.save(os.path.join(CACHE_DIR, f"{name}_val.npy"), val_emb)
    print(f"Saved {name} embeddings to {CACHE_DIR}")


def _load_cache(name):
    train_path = os.path.join(CACHE_DIR, f"{name}_train.npy")
    val_path = os.path.join(CACHE_DIR, f"{name}_val.npy")
    if os.path.exists(train_path) and os.path.exists(val_path):
        print(f"Loading cached {name} embeddings")
        return np.load(train_path), np.load(val_path)
    return None


def get_gemma_embeddings_v2(train_texts, val_texts):
    cached = _load_cache("gemma_v2")
    if cached: return cached

    model = SentenceTransformer("google/embeddinggemma-300m", device="cuda")
    train_emb = model.encode(list(train_texts), batch_size=256, prompt_name="Classification", show_progress_bar=True)
    val_emb = model.encode(list(val_texts), batch_size=256, prompt_name="Classification", show_progress_bar=True)
    del model; torch.cuda.empty_cache()

    _save_cache("gemma_v2", train_emb, val_emb)
    return train_emb, val_emb


# =============================================================================
# 2. ALTERNATIVE MODELS
# =============================================================================



def get_gte_multilingual_embeddings(train_texts, val_texts):
    cached = _load_cache("gte_multilingual")
    if cached: return cached

    model = SentenceTransformer("Alibaba-NLP/gte-multilingual-base", trust_remote_code=True, device="cuda")
    train_emb = model.encode(list(train_texts), batch_size=256, show_progress_bar=True)
    val_emb = model.encode(list(val_texts), batch_size=256, show_progress_bar=True)
    del model; torch.cuda.empty_cache()

    _save_cache("gte_multilingual", train_emb, val_emb)
    return train_emb, val_emb


def get_bge_m3_embeddings(train_texts, val_texts):
    cached = _load_cache("bge_m3")
    if cached: return cached

    model = SentenceTransformer("BAAI/bge-m3", device="cuda")
    train_emb = model.encode(list(train_texts), batch_size=256, show_progress_bar=True)
    val_emb = model.encode(list(val_texts), batch_size=256, show_progress_bar=True)
    del model; torch.cuda.empty_cache()

    _save_cache("bge_m3", train_emb, val_emb)
    return train_emb, val_emb


# =============================================================================
# 3. FINE-TUNING
# =============================================================================

def finetune_gemma(train_df, text_col="text", label_col="label",
                   output_dir="./gemma-finetuned",
                   epochs=3, batch_size=64, lr=2e-5):
    """
    Fine-tune EmbeddingGemma on your classification labels.
    Run ONCE, then use get_gemma_finetuned_embeddings.
    """
    if os.path.exists(output_dir) and os.listdir(output_dir):
        print(f"Fine-tuned model already exists at {output_dir}, skipping.")
        return

    model = SentenceTransformer("google/embeddinggemma-300m", device="cuda")

    train_dataset = Dataset.from_dict({
        "sentence": list(train_df[text_col]),
        "label": list(train_df[label_col].astype(int)),
    })

    loss = losses.BatchAllTripletLoss(model)

    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        warmup_ratio=0.1,
        fp16=True,
        logging_steps=100,
        save_strategy="epoch",
    )

    trainer = SentenceTransformerTrainer(
        model=model, args=args, train_dataset=train_dataset, loss=loss,
    )
    trainer.train()
    model.save_pretrained(output_dir)
    print(f"Fine-tuned model saved to {output_dir}")
    del model; torch.cuda.empty_cache()


def get_gemma_finetuned_embeddings(train_texts, val_texts, model_dir="./gemma-finetuned"):
    """Use after calling finetune_gemma()."""
    cached = _load_cache("gemma_finetuned")
    if cached: return cached

    model = SentenceTransformer(model_dir, device="cuda")
    train_emb = model.encode(list(train_texts), batch_size=256, prompt_name="Classification", show_progress_bar=True)
    val_emb = model.encode(list(val_texts), batch_size=256, prompt_name="Classification", show_progress_bar=True)
    del model; torch.cuda.empty_cache()

    _save_cache("gemma_finetuned", train_emb, val_emb)
    return train_emb, val_emb


