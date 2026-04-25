"""
Gemma_LogReg baseline model.

Combines:
  - Preprocessing: v4 (minimal - for transformers)
  - Embedding: Gemma seq128 (with caching)
  - Classifier: Logistic Regression
  
Validation Score (from notebook): 0.8948
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
import sys
from pathlib import Path

# Add parent directories to path
baselines_dir = Path(__file__).parent.parent
repo_root = baselines_dir.parent
sys.path.insert(0, str(baselines_dir))
sys.path.insert(0, str(repo_root))

from baseline_model import BaselineModel
from Resources.embeddings_adv import get_gemma_embeddings_seq128
from utils.preprocessing import preprocess


class GemmaLogReg(BaselineModel):
    """
    Gemma seq128 embeddings + Logistic Regression baseline.
    Uses v4 (minimal) preprocessing - best for transformer-based embedders.
    """
    
    def __init__(self):
        super().__init__()
    
    def get_model_name(self) -> str:
        return "Gemma_LogReg"
    
    def preprocess_text(self, series: pd.Series) -> pd.Series:
        """Apply v4 (minimal) preprocessing - preserves natural text for Gemma."""
        return series.apply(lambda x: preprocess(x, version=4))
    
    def vectorize(self, train_texts: pd.Series, val_texts: pd.Series) -> tuple:
        """
        Get Gemma seq128 embeddings with automatic caching.
        Falls back to CPU if CUDA is unavailable.
        """
        print("  Computing Gemma seq128 embeddings (cached)...")
        try:
            X_train, X_val = get_gemma_embeddings_seq128(
                train_texts.values, 
                val_texts.values
            )
        except Exception as e:
            if "CUDA" in str(e) or "cuda" in str(e).lower() or "GPU" in str(e):
                print(f"  ⚠ GPU error encountered, but embeddings should fall back to CPU automatically.")
                print(f"  Error details: {type(e).__name__}: {str(e)[:100]}")
                # The embeddings_adv.py should handle CUDA fallback internally
                # If we still get here, re-raise with more context
                raise RuntimeError(f"Embedding generation failed even after CUDA fallback: {e}") from e
            else:
                raise
        return X_train, X_val
    
    def get_model(self):
        """Factory method for fresh LogisticRegression instance."""
        return LogisticRegression(C=1.0, max_iter=1000)
