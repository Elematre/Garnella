"""
BoW_LogReg baseline model.

Combines:
  - Preprocessing: v1 (aggressive)
  - Embedding: Bag of Words (unigrams + bigrams)
  - Classifier: Logistic Regression
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import sys
from pathlib import Path

# Add parent directories to path
baselines_dir = Path(__file__).parent.parent
repo_root = baselines_dir.parent
sys.path.insert(0, str(baselines_dir))
sys.path.insert(0, str(repo_root))

from baseline_model import BaselineModel
from utils.preprocessing import preprocess


class BoWLogReg(BaselineModel):
    """
    Bag of Words + Logistic Regression baseline.
    """
    
    def __init__(self):
        super().__init__()
        self.vectorizer = None
    
    def get_model_name(self) -> str:
        return "BoW_LogReg"
    
    def preprocess_text(self, series: pd.Series) -> pd.Series:
        """Apply v1 (aggressive) preprocessing."""
        return series.apply(lambda x: preprocess(x, version=1))
    
    def vectorize(self, train_texts: pd.Series, val_texts: pd.Series) -> tuple:
        """
        Create Bag of Words embeddings.
        Fit only on training data, then transform both.
        """
        # Initialize vectorizer: unigrams + bigrams, max 10k features
        self.vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=10000)
        
        # Fit on training data only
        X_train = self.vectorizer.fit_transform(train_texts)
        
        # Transform validation data
        X_val = self.vectorizer.transform(val_texts)
        
        return X_train, X_val
    
    def get_model(self):
        """Factory method for fresh LogisticRegression instance."""
        return LogisticRegression(C=1.0, max_iter=1000)
