import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.trainer import SentenceTransformerTrainer
from datasets import Dataset
from peft import LoraConfig, get_peft_model

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


def _encode_with(model_name, cache_key, train_texts, val_texts,
                 prompt_name=None, trust_remote_code=False,
                 max_seq_length=128, batch_size=128):
    cached = _load_cache(cache_key)
    if cached is not None:
        return cached

    model = SentenceTransformer(
        model_name, device="cuda", trust_remote_code=trust_remote_code
    )
    model.max_seq_length = max_seq_length

    kwargs = {"batch_size": batch_size, "show_progress_bar": True}
    if prompt_name:
        kwargs["prompt_name"] = prompt_name

    train_emb = model.encode(list(train_texts), **kwargs)
    val_emb = model.encode(list(val_texts), **kwargs)

    del model
    torch.cuda.empty_cache()
    _save_cache(cache_key, train_emb, val_emb)
    return train_emb, val_emb


# =============================================================================
# 1. BASELINE EMBEDDINGS
# =============================================================================

def get_gemma_embeddings_v2(train_texts, val_texts):
    return _encode_with(
        "google/embeddinggemma-300m", "gemma_v2",
        train_texts, val_texts, prompt_name="Classification",
    )


# =============================================================================
# 2. ALTERNATIVE MODELS
# =============================================================================

def get_gte_multilingual_embeddings(train_texts, val_texts):
    return _encode_with(
        "Alibaba-NLP/gte-multilingual-base", "gte_multilingual",
        train_texts, val_texts, trust_remote_code=True,
    )


def get_bge_m3_embeddings(train_texts, val_texts):
    return _encode_with(
        "BAAI/bge-m3", "bge_m3", train_texts, val_texts,
    )


# =============================================================================
# 3. FINE-TUNING
# =============================================================================
def finetune_gemma(train_df, text_col="sentence", label_col="label",
                   output_dir="./gemma-finetuned",
                   epochs=2, batch_size=128, lr=2e-4,
                   max_seq_length=128):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    model = SentenceTransformer("google/embeddinggemma-300m", device="cuda")
    model.max_seq_length = max_seq_length

    # LoRA on attention projections
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules="all-linear",
        lora_dropout=0.1,
        bias="none",
    )
    peft_model = get_peft_model(model[0].auto_model, lora_config)
    peft_model.print_trainable_parameters()
    peft_model.gradient_checkpointing_enable()
    model[0].auto_model = peft_model

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
        bf16=True,
        logging_steps=50,
        save_strategy="epoch",
        dataloader_num_workers=0,
        report_to="none",
    )

    trainer = SentenceTransformerTrainer(
        model=model, args=args, train_dataset=train_dataset, loss=loss,
    )
    trainer.train()

    # Merge LoRA weights so the saved model is a standalone ST model
    # Grab the PEFT-wrapped model from the trainer (the SentenceTransformer
    # wrapper may have unwrapped auto_model along the way)
    trainer_auto_model = trainer.model[0].auto_model
    if hasattr(trainer_auto_model, "merge_and_unload"):
        model[0].auto_model = trainer_auto_model.merge_and_unload()
    else:
        model[0].auto_model = trainer_auto_model
    model.save_pretrained(output_dir)

    del model
    torch.cuda.empty_cache()


def get_gemma_finetuned_embeddings(train_texts, val_texts,
                                   model_dir="./gemma-finetuned"):
    """Use after calling finetune_gemma()."""
    return _encode_with(
        model_dir, "gemma_finetuned",
        train_texts, val_texts, prompt_name="Classification",
    )