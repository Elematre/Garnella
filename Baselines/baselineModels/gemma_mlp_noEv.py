"""
Gemma_MLP_v2 baseline model.

Combines:
  - Preprocessing: v4 (minimal - for transformers)
  - Embedding: Gemma seq128 (with caching)
  - Classifier: MLPClassifier-v2 (regularized, early stopping)
  
Validation Score (from notebook): 0.8974
"""

import pandas as pd
from sklearn.neural_network import MLPClassifier
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


class GemmaMLP_NoEV(BaselineModel):
    """
    Gemma seq128 embeddings + MLPClassifier-v2 baseline.
    Uses v4 (minimal) preprocessing - best for transformer-based embedders.
    
    MLPClassifier-v2 hyperparameters:
    - hidden_layer_sizes=(128,): single hidden layer with 128 units
    - alpha=1e-2: L2 regularization strength
    - early_stopping=True: stop when validation score plateaus
    - validation_fraction=0.1: use 10% of train data for early stopping validation
    - n_iter_no_change=10: stop if no improvement for 10 consecutive iterations
    - max_iter=300: maximum epochs
    - random_state=1: reproducibility
    """
    
    def __init__(self):
        super().__init__()
    
    def get_model_name(self) -> str:
        return "Gemma_MLP_NoEV"
    
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
        """Factory method for fresh MLPClassifier instance."""
        return MLPClassifier(
            hidden_layer_sizes=(128,),
            alpha=1e-2,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            max_iter=300,
            random_state=1
        )
