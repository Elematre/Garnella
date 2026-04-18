# The two tracks of the sentiment classification project

The project attacks the sentiment classification task in two fundamentally different ways.
Explanation i got from claude after being confused about my own code lol. 

## Track 1: Feature extraction + classifier

The core idea is that you turn each review into a fixed vector, then train a simple classifier on top of those vectors. The "turning text into vectors" step and the "classifying the vectors" step are separate — you do them one after another, and the vectorizer itself never changes during training.

You can think of it as a two-stage pipeline:

1. **Text → vector** (the "embedding" step)
2. **Vector → label** (the "classifier" step)

Because the vectors never change, you can compute them once, cache them to disk, and then try a bunch of different classifiers on the same cached vectors. That's why our `embedding_cache` exists and why the workflow iterates over `(embedding_fn, classifier_fn)` pairs.

### The models in Track 1

**Vectorizers (text → vector):**

- *TF-IDF* — counts how often words appear, weighted by how rare they are across documents. No semantic understanding; just surface word statistics.
- *Multilingual sentence encoder / Gemma* — pretrained language models used only to produce a fixed vector per review. We don't touch their weights; we just run each review through the model once and grab the output embedding.

**Classifiers (vector → label):**

- *Logistic regression* — linear classifier, fast, strong baseline
- *Linear SVM* — similar to LR, different loss function
- *MLP* — small neural network on top of the vectors, captures non-linear patterns
- *KNN* — predicts based on the k nearest training vectors
- *Random Forest / XGBoost* — tree ensembles, handle non-linearity differently than MLPs
- *Ridge regression / MLP regressor / XGBoost regressor* — same idea, but trained as regressors because the target is ordinal (0–4 stars). Round the output at prediction time.

**Decoding tricks on top (not separate models):**

- *Argmax* — pick the class with highest predicted probability (standard)
- *Expected-value decoding* — take E[y] = Σ i·p_i over the probabilities, then round. Better for the MAE metric because it uses the full probability distribution.

## Track 2: End-to-end fine-tuning

Here there's no separation between "embedding" and "classifier." You take a pretrained transformer (XLM-RoBERTa, mDeBERTa, etc.), stick a small classification head on top, and train the whole thing — encoder weights included — on our labeled data. The gradients from the classification loss flow all the way back through the transformer, reshaping its internal representations to be good at our task specifically.

Because of this, there's no caching: every run trains a full model from pretrained starting point to sentiment-tuned model.

### The models in Track 2

- *XLM-RoBERTa base* — strong default, handles EN+DE natively, ~280M params
- *XLM-RoBERTa large* — same family, bigger, slower to train, usually 1–2 points better
- *mDeBERTa v3 base* — often beats XLM-R base at similar size
- *cardiffnlp/twitter-xlm-roberta-base-sentiment* — already pretrained on sentiment data, fewer epochs needed

## Why have both tracks?

This is the part to really internalize, because it's what the report is built on.

**Track 1 gives us breadth.** Many models, many rows, clear comparisons. Shows we explored the design space methodically. Cheap per-row because of caching.

**Track 2 gives us points.** Fine-tuning almost always beats frozen embeddings on a supervised task with 252k labels. This is what clears the hard baseline (0.906). Expensive per-row — each run is hours.

We need both. Track 1 alone won't hit 0.906. Track 2 alone gives us no ablation story — just "we fine-tuned XLM-R and got 0.91, the end," which reads as thin in a 50%-weighted report. The strongest submission has Track 1 baselines showing a methodical progression from TF-IDF up through Gemma embeddings, and then Track 2 on top showing the fine-tuning jump — with a clear narrative about why each step helps.

## Key technical difference

If someone asks "why can't we just fine-tune in Track 1?" — the answer is that Track 1's whole structure depends on the embedding being a fixed, cacheable function. Once you allow the embedding to change during training, the cache is invalid and the `(embedding, classifier)` pairing pattern breaks. Fine-tuning is fundamentally a different computational shape, not just a different model choice. That's why it lives in its own module (`finetune.py`).

## Summary table

| | Track 1 | Track 2 |
|---|---|---|
| What changes during training | Only the classifier | Encoder + classifier (everything) |
| Cacheable? | Yes, embeddings reused across classifiers | No, full retrain every time |
| Cost per run | Seconds to minutes | Hours |
| Expected score range | 0.87–0.90 | 0.90–0.92 |
| Role in report | Breadth, baselines, ablations | Points, final submission |

