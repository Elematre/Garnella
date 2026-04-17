import numpy as np
import torch
import os
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.trainer import SentenceTransformerTrainer
from datasets import Dataset



def get_gemma_embeddings_v2(train_texts, val_texts):
    model = SentenceTransformer("google/embeddinggemma-300m", device="cuda")
    train_emb = model.encode(list(train_texts), batch_size=256, prompt_name="Classification", show_progress_bar=True)
    val_emb = model.encode(list(val_texts), batch_size=256, prompt_name="Classification", show_progress_bar=True)
    del model; torch.cuda.empty_cache()
    return train_emb, val_emb

# ~305M params — good multilingual coverage (70+ languages)
def get_gte_multilingual_embeddings(train_texts, val_texts):
    model = SentenceTransformer("Alibaba-NLP/gte-multilingual-base", trust_remote_code=True, device="cuda")
    train_emb = model.encode(list(train_texts), batch_size=256, show_progress_bar=True)
    val_emb = model.encode(list(val_texts), batch_size=256, show_progress_bar=True)
    del model; torch.cuda.empty_cache()
    return train_emb, val_emb

# ~568M params — might be too large, but best multilingual open model
def get_bge_m3_embeddings(train_texts, val_texts):
    model = SentenceTransformer("BAAI/bge-m3", device="cuda")
    train_emb = model.encode(list(train_texts), batch_size=256, show_progress_bar=True)
    val_emb = model.encode(list(val_texts), batch_size=256, show_progress_bar=True)
    del model; torch.cuda.empty_cache()
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

    BatchAllTripletLoss pulls same-class texts together and pushes
    different-class texts apart. With 252k samples and 3-10 classes
    this works well out of the box.
    """
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
    model = SentenceTransformer(model_dir, device="cuda")
    train_emb = model.encode(list(train_texts), batch_size=256, prompt_name="Classification", show_progress_bar=True)
    val_emb = model.encode(list(val_texts), batch_size=256, prompt_name="Classification", show_progress_bar=True)
    del model; torch.cuda.empty_cache()
    return train_emb, val_emb


