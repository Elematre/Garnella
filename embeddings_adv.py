import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction
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
                 max_seq_length=64, batch_size=512):
    cached = _load_cache(cache_key)
    if cached is not None:
        return cached

    model = SentenceTransformer(
        model_name, device="cuda", trust_remote_code=trust_remote_code,
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


def get_gemma_embeddings_v2(train_texts, val_texts):
    return _encode_with(
        "google/embeddinggemma-300m", "gemma_v2",
        train_texts, val_texts, prompt_name="Classification",
    )




def get_gemma_embeddings_seq128(train_texts, val_texts):
    return _encode_with(
        "google/embeddinggemma-300m", "gemma_seq128",
        train_texts, val_texts, prompt_name="Classification",
        max_seq_length=128,
    )

def get_gemma_embeddings_seq256(train_texts, val_texts):
    return _encode_with(
        "google/embeddinggemma-300m", "gemma_seq256",
        train_texts, val_texts, prompt_name="Classification",
        max_seq_length=256,
    )


# =============================================================================
# Sentiment-pretrained frozen embedders (Track 1)
# =============================================================================
# These are transformers that someone else fine-tuned on sentiment tasks.
# We use them as frozen feature extractors — no training on our end. Their
# embedding space is organized around sentiment rather than topical similarity,
# so a simple classifier on top should outperform general-purpose embedders
# like Gemma for this task.

from transformers import AutoTokenizer, AutoModel

def _mean_pool(last_hidden_state, attention_mask):
    """Attention-masked mean pooling over token dimension."""
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def _encode_hf(model_name, cache_key, train_texts, val_texts,
               max_seq_length=128, batch_size=256, pooling="mean"):
    """Run a HuggingFace transformer forward-only over the texts and cache the output.

    pooling: "mean" (masked mean over tokens) or "cls" (first token).
    Mean pooling is the safer default for sentiment embedders; CLS is what
    classification-head models like cardiffnlp/twitter-xlm-roberta use during
    fine-tuning, so it's also a reasonable choice for that specific model.
    """
    cached = _load_cache(cache_key)
    if cached is not None:
        return cached

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
    model.eval()

    def encode(texts):
        texts = list(texts)
        out = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = texts[start:start + batch_size]
                enc = tokenizer(batch, padding=True, truncation=True,
                                max_length=max_seq_length, return_tensors="pt").to(device)
                hidden = model(**enc).last_hidden_state
                if pooling == "cls":
                    vecs = hidden[:, 0, :]
                else:
                    vecs = _mean_pool(hidden, enc["attention_mask"])
                # Cast to fp32 before moving to numpy (bfloat16 → numpy conversion needs this)
                out.append(vecs.float().cpu().numpy())
                if start % (batch_size * 20) == 0:
                    print(f"  {cache_key}: encoded {start + len(batch)}/{len(texts)}")
        return np.vstack(out)

    train_emb = encode(train_texts)
    val_emb = encode(val_texts)

    del model, tokenizer
    torch.cuda.empty_cache()

    _save_cache(cache_key, train_emb, val_emb)
    return train_emb, val_emb

def get_nlptown_sentiment_embeddings(train_texts, val_texts):
    """nlptown/bert-base-multilingual-uncased-sentiment as frozen feature extractor.

    BERT-multilingual fine-tuned on product reviews (Amazon) for 1-5 star
    prediction. This is the closest possible label-scheme match to our task
    — literally the same 5-star prediction, on product reviews, covering both
    English and German. Uses WordPiece tokenization so no sentencepiece
    dependency.
    """
    return _encode_hf(
        "nlptown/bert-base-multilingual-uncased-sentiment",
        "nlptown_sentiment",
        train_texts, val_texts,
        max_seq_length=128, batch_size=512, pooling="cls",
    )


def get_multilingual_e5_embeddings(train_texts, val_texts):
    """intfloat/multilingual-e5-base as frozen feature extractor.

    Not sentiment-specialized — it's a general-purpose multilingual embedder
    trained with a contrastive objective for retrieval/similarity tasks.
    Included as a comparison point: is a high-quality general embedder
    competitive with sentiment-specialized ones? E5 requires the 'query: '
    prefix by convention, but for classification we use 'passage: ' which
    matches how it was trained on document-side inputs.
    """
    prefixed_train = [f"passage: {t}" for t in train_texts]
    prefixed_val = [f"passage: {t}" for t in val_texts]
    return _encode_hf(
        "intfloat/multilingual-e5-base",
        "multilingual_e5",
        prefixed_train, prefixed_val,
        max_seq_length=128, batch_size=512, pooling="mean",  # E5 uses mean pooling
    )

def get_tabularisai_sentiment_embeddings(train_texts, val_texts):
    """tabularisai/multilingual-sentiment-analysis — another sentiment-pretrained
    multilingual model. Fine-tuned on reviews, so it may be closer in domain
    to our product-review data than the Twitter model.
    """
    return _encode_hf(
        "tabularisai/multilingual-sentiment-analysis",
        "tabularisai_sentiment",
        train_texts, val_texts,
        max_seq_length=128, batch_size=512, pooling="cls",
    )


# =============================================================================
#  FINE-TUNING GEMMA (ordinal-aware via CoSENT) --> UNNECESSARY
# =============================================================================

# def _build_cosent_pairs(texts, labels, n_pairs=150_000, seed=1):
#     """
#     Build (sentence1, sentence2, score) triples where score reflects ordinal
#     closeness: 1.0 for same class, decaying linearly with |label_i - label_j|.
#     CoSENT only uses the *ranking* of scores within a batch, so the exact
#     scale doesn't matter — only that closer labels → higher score.
#     """
#     rng = np.random.default_rng(seed)
#     texts = np.asarray(texts)
#     labels = np.asarray(labels, dtype=float)
#     n = len(texts)

#     label_range = labels.max() - labels.min()
#     if label_range == 0:
#         label_range = 1.0

#     i = rng.integers(0, n, size=n_pairs)
#     j = rng.integers(0, n, size=n_pairs)
#     mask = i != j
#     i, j = i[mask], j[mask]

#     scores = 1.0 - np.abs(labels[i] - labels[j]) / label_range
#     return {
#         "sentence1": texts[i].tolist(),
#         "sentence2": texts[j].tolist(),
#         "score": scores.astype(np.float32).tolist(),
#     }


# def finetune_gemma(train_df, val_df=None,
#                    text_col="sentence", label_col="label",
#                    output_dir="./gemma-finetuned",
#                    epochs=3, batch_size=128, lr=2e-4,
#                    max_seq_length=64, n_train_pairs=150_000,
#                    n_eval_pairs=5_000):
#     os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

#     model = SentenceTransformer("google/embeddinggemma-300m", device="cuda")
#     model.max_seq_length = max_seq_length

#     lora_config = LoraConfig(
#         r=16,
#         lora_alpha=32,
#         target_modules="all-linear",
#         lora_dropout=0.1,
#         bias="none",
#     )
#     peft_model = get_peft_model(model[0].auto_model, lora_config)
#     peft_model.print_trainable_parameters()
#     model[0].auto_model = peft_model

#     train_pairs = _build_cosent_pairs(
#         train_df[text_col].tolist(),
#         train_df[label_col].astype(float).tolist(),
#         n_pairs=n_train_pairs, seed=1,
#     )
#     train_dataset = Dataset.from_dict(train_pairs)

#     evaluator = None
#     if val_df is not None:
#         eval_pairs = _build_cosent_pairs(
#             val_df[text_col].tolist(),
#             val_df[label_col].astype(float).tolist(),
#             n_pairs=n_eval_pairs, seed=2,
#         )
#         evaluator = EmbeddingSimilarityEvaluator(
#             sentences1=eval_pairs["sentence1"],
#             sentences2=eval_pairs["sentence2"],
#             scores=eval_pairs["score"],
#             main_similarity=SimilarityFunction.COSINE,
#             name="ordinal-val",
#         )

#     loss = losses.CoSENTLoss(model)

#     args = SentenceTransformerTrainingArguments(
#         output_dir=output_dir,
#         num_train_epochs=epochs,
#         per_device_train_batch_size=batch_size,
#         learning_rate=lr,
#         warmup_ratio=0.1,
#         bf16=True,
#         logging_steps=50,
#         eval_strategy="epoch" if evaluator is not None else "no",
#         save_strategy="epoch",
#         load_best_model_at_end=evaluator is not None,
#         metric_for_best_model="eval_ordinal-val_spearman_cosine",
#         greater_is_better=True,
#         save_total_limit=2,
#         dataloader_num_workers=0,
#         report_to="none",
#     )

#     trainer = SentenceTransformerTrainer(
#         model=model, args=args,
#         train_dataset=train_dataset,
#         evaluator=evaluator,
#         loss=loss,
#     )
#     trainer.train()

#     trainer_auto_model = trainer.model[0].auto_model
#     if hasattr(trainer_auto_model, "merge_and_unload"):
#         model[0].auto_model = trainer_auto_model.merge_and_unload()
#     else:
#         model[0].auto_model = trainer_auto_model

#     model.save(output_dir)

#     del model
#     torch.cuda.empty_cache()
#     reloaded = SentenceTransformer(output_dir, device="cuda")
#     _ = reloaded.encode(["sanity check"], show_progress_bar=False)
#     del reloaded
#     torch.cuda.empty_cache()
#     print(f"Saved and verified fine-tuned model at {output_dir}")


# def get_gemma_finetuned_embeddings(train_texts, val_texts,
#                                    model_dir="./gemma-finetuned"):
#     """Use after calling finetune_gemma()."""
#     return _encode_with(
#         model_dir, "gemma_finetuned",
#         train_texts, val_texts, prompt_name="Classification",
#     )