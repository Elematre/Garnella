"""
Preprocessing pipeline for the CIL sentiment classification project.

Six versions — v1/v2 flatten title+body into one string, v3/v4/v5/v6 preserve structure:
  v1 - Aggressive         : lowercase, strip HTML, convert emojis, normalize repeated chars,
                            remove punctuation, remove numbers, remove stopwords (DE+EN)
                            title+body joined with a space, \n\n gone
  v2 - Moderate           : same as v1 but keeps ! and ?
                            title+body joined with a space, \n\n gone
  v3 - Light              : lowercase, strip HTML, convert emojis, normalize repeated chars
                            keeps punctuation, numbers, stopwords — \n\n replaced with [SEP]
  v4 - Minimal            : strip HTML only, \n\n preserved
                            best for transformers that do their own tokenization
  v5 - Aggressive+struct  : same cleaning as v1, but title and body cleaned separately
                            and rejoined with \n\n — structure preserved
  v6 - Moderate+struct    : same cleaning as v2, but title and body cleaned separately
                            and rejoined with \n\n — structure preserved

Usage:
    from preprocessing import preprocess_df
    train = pd.read_csv("data/train.csv")
    train["text_v1"] = preprocess_df(train["sentence"], version=1)
    train["text_v4"] = preprocess_df(train["sentence"], version=4)
"""

import re
import unicodedata
import pandas as pd

# ---------------------------------------------------------------------------
# Emoji → sentiment word map
# Unknown emojis are stripped (non-BMP unicode removed as fallback).
# ---------------------------------------------------------------------------
EMOJI_MAP = {
    "😊": " happy ",    "😄": " happy ",    "😀": " happy ",
    "😍": " love ",     "❤️": " love ",     "🥰": " love ",
    "😢": " sad ",      "😭": " sad ",
    "😡": " angry ",    "😠": " angry ",    "😤": " frustrated ",
    "👍": " good ",     "👍🏼": " good ",    "👍🏻": " good ",
    "👎": " bad ",      "👎🏻": " bad ",
    "⭐": " star ",     "🌟": " great ",    "🔥": " great ",
    "💩": " bad ",      "💔": " heartbreak ",
    "😐": " neutral ",  "🤷": " neutral ",
    "😒": " disappointed ", "🙄": " annoyed ", "😑": " annoyed ",
    "😳": " shocked ",
}

# ---------------------------------------------------------------------------
# Optional: nltk stopwords (install with: pip install nltk)
# Falls back to a small built-in list if nltk is not available.
# ---------------------------------------------------------------------------
_NEGATIONS_EN = {"not", "no", "nor", "never", "neither", "nobody", "nothing", "nowhere"}
_NEGATIONS_DE = {"nicht", "kein", "keine", "keinen", "keinem", "keiner", "niemals", "nie", "nichts"}

try:
    import nltk
    from nltk.corpus import stopwords
    try:
        _SW_EN = set(stopwords.words("english")) - _NEGATIONS_EN
        _SW_DE = set(stopwords.words("german")) - _NEGATIONS_DE
    except LookupError:
        nltk.download("stopwords", quiet=True)
        _SW_EN = set(stopwords.words("english")) - _NEGATIONS_EN
        _SW_DE = set(stopwords.words("german")) - _NEGATIONS_DE

except ImportError:
    # Minimal fallback stopword lists
    _SW_EN = {
        "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
        "they", "them", "the", "a", "an", "is", "are", "was", "were",
        "be", "been", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "shall", "can",
        "and", "or", "but", "if", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "this", "that",
    }
    _SW_DE = {
        "ich", "du", "er", "sie", "es", "wir", "ihr", "sie", "mich",
        "mir", "mein", "dich", "dir", "dein", "ihn", "ihm", "sein",
        "uns", "unser", "euch", "euer", "die", "der", "das", "den",
        "dem", "des", "ein", "eine", "einen", "einem", "einer", "eines",
        "und", "oder", "aber", "wenn", "in", "auf", "an", "zu", "für",
        "von", "mit", "aus", "bei", "nach", "ist", "sind", "war", "waren",
        "hat", "haben", "hatte", "hatten", "wird", "werden", "kann",
        "auch", "noch", "schon", "sehr", "so", "wie", "als",
    }

_STOPWORDS = _SW_EN | _SW_DE


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _split_title_body(text: str) -> tuple[str, str]:
    """Split title from body on the blank line separating them."""
    parts = text.split("\n\n", 1)
    title = parts[0].strip()
    body = parts[1].strip() if len(parts) > 1 else ""
    return title, body


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text)


def _emojis_to_text(text: str) -> str:
    """Replace known emojis with sentiment words via EMOJI_MAP.
    Unknown emojis are stripped."""
    for emoji, replacement in EMOJI_MAP.items():
        text = text.replace(emoji, replacement)
    # Strip any remaining emoji / non-BMP characters not in the map
    text = "".join(
        ch for ch in text
        if unicodedata.category(ch) not in ("So", "Cs", "Co", "Cn")
    )
    return text


def _normalize_repeated_chars(text: str) -> str:
    """sooooo → soo  (keep max 2 repeated chars to preserve emphasis signal)."""
    return re.sub(r"(.)\1{2,}", r"\1\1", text)


def _remove_punctuation(text: str) -> str:
    return re.sub(r"[^\w\s]", " ", text)


def _remove_punctuation_keep_sentiment(text: str) -> str:
    """Keep ! and ? as they carry sentiment signal."""
    return re.sub(r"[^\w\s!?]", " ", text)


def _remove_numbers(text: str) -> str:
    return re.sub(r"\d+", " ", text)


def _remove_stopwords(text: str) -> str:
    return " ".join(w for w in text.split() if w not in _STOPWORDS)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalize_spaces(text: str) -> str:
    """Collapse runs of horizontal whitespace only — preserves newlines."""
    return re.sub(r"[^\S\n]+", " ", text).strip()


# ---------------------------------------------------------------------------
# Version implementations
# ---------------------------------------------------------------------------

def _v1(text: str) -> str:
    """
    Aggressive — good for BoW / TF-IDF with classical ML (logistic regression, SVM).
    Loses most surface-level noise
    """
    title, body = _split_title_body(text)
    # Combine with a separator so the model can still distinguish them if needed
    text = title + " " + body
    text = _strip_html(text)
    text = _emojis_to_text(text)
    text = text.lower()
    text = _normalize_repeated_chars(text)
    text = _remove_punctuation(text)
    text = _remove_numbers(text)
    text = _remove_stopwords(text)
    text = _normalize_whitespace(text)
    return text


def _v2(text: str) -> str:
    """
    Moderate — good for BoW / TF-IDF or simple neural models (MLP, CNN).
    Keeps ! and ? for sentiment signal; removes stopwords but not stemming.
    """
    title, body = _split_title_body(text)
    text = title + " " + body
    text = _strip_html(text)
    text = _emojis_to_text(text)
    text = text.lower()
    text = _normalize_repeated_chars(text)
    text = _remove_punctuation_keep_sentiment(text)
    text = _remove_numbers(text)
    text = _remove_stopwords(text)
    text = _normalize_whitespace(text)
    return text


def _v3(text: str) -> str:
    """
    Light — good for RNNs or fine-tuned models with their own tokenizer.
    Keeps numbers, most punctuation, and stopwords; just cleans noise.
    """
    title, body = _split_title_body(text)
    # Use [SEP] marker so models that read the full string can distinguish sections
    text = title + " [SEP] " + body if body else title
    text = _strip_html(text)
    text = _emojis_to_text(text)
    text = text.lower()
    text = _normalize_repeated_chars(text)
    text = _normalize_whitespace(text)
    return text


def _v4(text: str) -> str:
    """
    Minimal — intended for transformers like Gemma / BERT that do their own
    sub-word tokenization and benefit from seeing raw, natural text.
    Preserves the blank line between title and body so callers can split on \n\n
    and pass (title, body) as a sentence pair to the tokenizer if needed.
    """
    title, body = _split_title_body(text)
    text = title + "\n\n" + body if body else title
    text = _strip_html(text)          # still remove accidental HTML tags
    text = _normalize_spaces(text)
    return text


def _v5(text: str) -> str:
    """
    Aggressive cleaning (like v1) but preserves the \\n\\n title/body boundary.
    Use to test whether Gemma / transformer embeddings benefit from cleaned input.
    """
    title, body = _split_title_body(text)
    title = _strip_html(title)
    title = _emojis_to_text(title)
    title = title.lower()
    title = _normalize_repeated_chars(title)
    title = _remove_punctuation(title)
    title = _remove_numbers(title)
    title = _remove_stopwords(title)
    title = _normalize_spaces(title)
    body = _strip_html(body)
    body = _emojis_to_text(body)
    body = body.lower()
    body = _normalize_repeated_chars(body)
    body = _remove_punctuation(body)
    body = _remove_numbers(body)
    body = _remove_stopwords(body)
    body = _normalize_spaces(body)
    return title + "\n\n" + body if body else title


def _v6(text: str) -> str:
    """
    Moderate cleaning (like v2, keeps ! and ?) but preserves the \\n\\n title/body boundary.
    Use to test whether Gemma / transformer embeddings benefit from lightly cleaned input.
    """
    title, body = _split_title_body(text)
    title = _strip_html(title)
    title = _emojis_to_text(title)
    title = title.lower()
    title = _normalize_repeated_chars(title)
    title = _remove_punctuation_keep_sentiment(title)
    title = _remove_numbers(title)
    title = _remove_stopwords(title)
    title = _normalize_spaces(title)
    body = _strip_html(body)
    body = _emojis_to_text(body)
    body = body.lower()
    body = _normalize_repeated_chars(body)
    body = _remove_punctuation_keep_sentiment(body)
    body = _remove_numbers(body)
    body = _remove_stopwords(body)
    body = _normalize_spaces(body)
    return title + "\n\n" + body if body else title


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_VERSION_MAP = {1: _v1, 2: _v2, 3: _v3, 4: _v4, 5: _v5, 6: _v6}


def preprocess(text: str, version: int = 2) -> str:
    """
    Preprocess a single review string.

    Args:
        text:    Raw text from the 'sentence' column.
        version: 1 (aggressive) … 6 (moderate + structure preserved).
    """
    if version not in _VERSION_MAP:
        raise ValueError(f"version must be 1–6, got {version}")
    return _VERSION_MAP[version](text)


def preprocess_df(series: pd.Series, version: int = 2) -> pd.Series:
    """
    Preprocess a pandas Series of raw review strings.

    Args:
        series:  The 'sentence' column from train.csv / test.csv.
        version: 1 (aggressive) … 6 (moderate + structure preserved).

    Returns:
        A new Series with preprocessed strings.
    """
    fn = _VERSION_MAP.get(version)
    if fn is None:
        raise ValueError(f"version must be 1–6, got {version}")
    return series.apply(fn)


# ---------------------------------------------------------------------------
# Quick demo / sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    sample = (
        "Sooooo enttäuschend!!!!\n\n"
        "Ich habe dieses Produkt für 29,99 € gekauft und es ist <b>komplett kaputt</b> "
        "angekommen 😡😡. Würde es NIEMALS wieder kaufen!!! The worst thing ever."
    )

    for v in [1, 2, 3, 4, 5, 6]:
        print(f"\n--- Version {v} ---")
        print(preprocess(sample, version=v))

    # If a CSV path is passed, save one file per preprocessing version
    if len(sys.argv) > 1:
        path = sys.argv[1]
        df = pd.read_csv(path)
        for v in [1, 2, 3, 4,5,6]:
            out_df = df.copy()
            out_df["sentence"] = preprocess_df(df["sentence"], version=v)
            out = path.replace(".csv", f"_v{v}.csv")
            out_df.to_csv(out, index=False)
            print(f"Saved version {v} → {out}")
