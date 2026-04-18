# CIL Sentiment Classification - Preprocessing

All preprocessing logic lives in `preprocessing.py`. Run it from the command line to produce one CSV per version:

```bash
python preprocessing.py /path/to/data/train.csv
# → train_v1.csv, train_v2.csv, train_v3.csv, train_v4.csv, train_v5.csv, train_v6.csv
```

Or use it inside a notebook/script:

```python
from preprocessing import preprocess_df
import pandas as pd

train = pd.read_csv("data/train.csv")
train["sentence"] = preprocess_df(train["sentence"], version=2)
```

---

### Version 1 — Aggressive

**What it does:**
- Converts known emojis to sentiment words via a hardcoded map (😡 → `angry`)
- Lowercases everything
- Normalizes repeated characters (`sooooo` → `soo`)
- Removes all punctuation
- Removes numbers
- Removes stopwords (English + German)

**Structure:** title and body are joined with a space — the `\n\n` boundary is gone. All whitespace collapsed to single spaces.

**Result:** Very compact text, small vocabulary, loses nuance but generalizes well.

**Recommended for:** Bag-of-Words + Logistic Regression, Bag-of-Words + SVM

---

### Version 2 — Moderate

**What it does:**
- Converts known emojis to sentiment words 
- Lowercases everything
- Normalizes repeated characters
- Removes most punctuation but **keeps `!` and `?`** (strong sentiment signals)
- Removes numbers
- Removes stopwords (English + German)

**Structure:** same as v1 — title and body joined with a space, `\n\n` gone, all whitespace collapsed.

**Result:** Clean but readable text that still preserves some sentiment markers.

**Recommended for:** TF-IDF + Logistic Regression, TF-IDF + SVM, simple MLP

---

### Version 3 — Light

**What it does:**
- Converts known emojis to sentiment words
- Lowercases everything
- Normalizes repeated characters
- Strips HTML tags
- Keeps punctuation, numbers, and stopwords
- Splits title and body with a `[SEP]` token

**Structure:** `\n\n` replaced with `[SEP]`, all whitespace collapsed to single spaces — one flat string.

**Result:** Near-natural text, lowercased. The `[SEP]` marker lets models distinguish the review title from the body.

**Recommended for:** 1D CNN, Bidirectional RNN/LSTM, smaller transformer models

---

### Version 4 — Minimal

**What it does:**
- Strips HTML tags
- Preserves the original `\n\n` between title and body — structure intact, no flattening
- Preserves original casing, emojis, punctuation, numbers, and stopwords

**Result:** Almost raw text. Transformers handle their own sub-word tokenization and benefit from seeing natural language.

**Recommended for:** Gemma, BERT, or any pre-trained transformer model

---

### Version 5 — Aggressive + structure preserved

**What it does:**
- Same cleaning as v1 (emojis → words, lowercase, normalize repeated chars, remove punctuation, remove numbers, remove stopwords)
- But title and body are cleaned separately and rejoined with `\n\n`

**Structure:** `\n\n` boundary preserved — can still split on it for sentence-pair tokenization.

**Result:** Aggressively cleaned text but with the title/body boundary intact.

**Recommended for:** Testing whether Gemma embeddings improve with aggressive preprocessing while keeping structure

---

### Version 6 — Moderate + structure preserved

**What it does:**
- Same cleaning as v2 (emojis → words, lowercase, normalize repeated chars, remove most punctuation but **keeps `!` and `?`**, remove numbers, remove stopwords)
- But title and body are cleaned separately and rejoined with `\n\n`

**Structure:** `\n\n` boundary preserved — can still split on it for sentence-pair tokenization.

**Result:** Moderately cleaned text with the title/body boundary intact.

**Recommended for:** Testing whether Gemma embeddings improve with moderate preprocessing while keeping structure

