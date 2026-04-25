# The two tracks of the sentiment classification project

The project attacks the sentiment classification task in fundamentally different ways.
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
- *Multilingual sentence encoder / Gemma* — pretrained transformer models used only to produce a fixed vector per review. We don't touch their weights; we just run each review through the model once and grab the output embedding. These embeddings are organized around *topical similarity* (what the text is about) because that's what their contrastive training objective optimized for.
- *Sentiment-pretrained encoders* (e.g. `cardiffnlp/twitter-xlm-roberta-base-sentiment`, `tabularisai/multilingual-sentiment-analysis`) — transformers that someone else already fine-tuned on sentiment data. Used in our pipeline as frozen feature extractors, their embeddings are organized around *sentiment* rather than topic. Expected to outperform general-purpose embedders like Gemma on this task. Still Track 1 because we don't update their weights — the fine-tuning was done by someone else, on a different dataset.

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

## Track 1.5: Small neural nets trained end-to-end on word embeddings

CNNs and RNNs as described in the course slides don't fit cleanly into either track, so they deserve their own category.

The setup is: start from **token-level word embeddings** (GloVe, FastText, or a learnable `nn.Embedding` table), pad reviews to a fixed sequence length, and feed the resulting `(seq_len, embedding_dim)` tensor into a small neural network. The network itself is trained from scratch on our labeled data, but the word embeddings underneath can be frozen or trainable depending on choice.

### The models in Track 1.5

- *1D CNN on FastText embeddings* — the Yoon Kim 2014 architecture the slides show a figure of. Convolutional filters slide over tokens capturing local n-gram patterns ("not good", "worth the money"), max-pooled, fed into a classification head.
- *Bidirectional LSTM / GRU on FastText embeddings* — processes the sequence in both directions, better at long-range dependencies than CNNs.

**Why our cached Gemma embeddings don't directly work here.** Our Gemma cache stores *one pooled vector per review* (the embedding was averaged across tokens when we extracted it). CNNs and RNNs need the *sequence of token vectors* — the information about which word was in which position, which got collapsed during pooling. So Track 1.5 requires its own embedding setup, typically pretrained word embeddings like FastText.

**Role of Track 1.5 in the report.** The course slides explicitly list 1D CNN and Bidirectional RNN as suggested architectures, so including at least one row here shows we explored the classical deep-learning-for-NLP approaches. Expected performance is between Track 1 and Track 2 — better than frozen general-purpose embeddings because the CNN/RNN is trained on our task, but worse than fine-tuning a full pretrained transformer.

## Track 2: End-to-end fine-tuning of pretrained transformers

Here there's no separation between "embedding" and "classifier." You take a pretrained transformer (XLM-RoBERTa, mDeBERTa, etc.), stick a small classification head on top, and train the whole thing — encoder weights included — on our labeled data. The gradients from the classification loss flow all the way back through the transformer, reshaping its internal representations to be good at our task specifically.

Because of this, there's no caching: every run trains a full model from pretrained starting point to sentiment-tuned model.

### The models in Track 2

- *XLM-RoBERTa base* — strong default, handles EN+DE natively, ~280M params
- *XLM-RoBERTa large* — same family, bigger, slower to train, usually 1–2 points better
- *mDeBERTa v3 base* — often beats XLM-R base at similar size
- *cardiffnlp/twitter-xlm-roberta-base-sentiment* — already pretrained on sentiment data, fewer epochs needed. Note: the same model shows up in Track 1 as a frozen feature extractor; the difference is whether we update its weights during training.



## Key technical distinction: what gets updated during training?

The three tracks differ in *what* gets trained, not in *which architecture* is used:

- **Track 1:** only the classifier. Feature extractor (even if it's a large transformer like Gemma) is frozen.
- **Track 1.5:** a small neural network (CNN/RNN) trained from scratch, typically on frozen pretrained word embeddings.
- **Track 2:** the entire pretrained transformer plus the classification head, trained jointly on our labels.

The same architecture can live in different tracks depending on usage — a transformer used for inference-only feature extraction is Track 1, the same transformer fine-tuned on our labels is Track 2.

If someone asks "why can't we just fine-tune in Track 1?" — the answer is that Track 1's whole structure depends on the embedding being a fixed, cacheable function. Once you allow the embedding to change during training, the cache is invalid and the `(embedding, classifier)` pairing pattern breaks. Fine-tuning is fundamentally a different computational shape, not just a different model choice. That's why it lives in its own module (`finetune.py`).

## Summary table

| | Track 1 | Track 1.5 | Track 2 |
|---|---|---|---|
| What changes during training | Only the classifier | Small neural net (CNN/RNN) trained from scratch | Encoder + classifier (everything) |
| Starting point | Frozen pretrained embeddings (Gemma, TF-IDF, sentiment-XLM-R) | Frozen word embeddings (FastText / GloVe) | Pretrained transformer (XLM-R, mDeBERTa) |
| Cacheable? | Yes, embeddings reused across classifiers | Partially — word embeddings can be cached, the CNN/RNN itself retrains | No, full retrain every time |
| Cost per run | Seconds to minutes | Minutes to ~1 hour | Hours |
| Expected score range | 0.87–0.90 (higher with sentiment-pretrained embedder) | 0.88–0.91 | 0.90–0.92 |
| Role in report | Breadth, baselines, ablations | Classical deep-NLP coverage | Points, final submission |