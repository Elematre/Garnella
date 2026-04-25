"""
Abstract base class for baseline models.

Each baseline encapsulates:
  - Preprocessing strategy (version)
  - Embedding function (vectorizer)
  - Classifier model (factory function)
  - Training and evaluation logic
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


class BaselineModel(ABC):
    """
    Abstract base class for sentiment classification baselines.
    
    Subclasses implement specific (embedding, classifier) combinations.
    """
    
    def __init__(self):
        self.train_data = None
        self.val_data = None
        self.X_train = None
        self.X_val = None
        self.Y_train = None
        self.Y_val = None
        self.model = None
        self.results = {}
    
    @abstractmethod
    def get_model_name(self) -> str:
        """
        Return the model identifier (e.g., "BoW_LogReg").
        """
        pass
    
    @abstractmethod
    def preprocess_text(self, series: pd.Series) -> pd.Series:
        """
        Apply preprocessing to a text series.
        
        Args:
            series: Text data to preprocess
            
        Returns:
            Preprocessed text series
        """
        pass
    
    @abstractmethod
    def vectorize(self, train_texts: pd.Series, val_texts: pd.Series) -> tuple:
        """
        Vectorize training and validation texts.
        Fit vectorizer only on training data.
        
        Args:
            train_texts: Training text data
            val_texts: Validation text data
            
        Returns:
            (X_train, X_val) - vectorized/embedded representations
        """
        pass
    
    @abstractmethod
    def get_model(self):
        """
        Factory method to create a fresh, untrained model instance.
        
        Returns:
            Untrained classifier model
        """
        pass
    
    def load_and_preprocess_data(self, train_path: str, train_size: float = 0.9, random_state: int = 42):
        """
        Load data, apply preprocessing, and split into train/val.
        
        Args:
            train_path: Path to training CSV
            train_size: Fraction for training set (rest goes to validation)
            random_state: Random seed for reproducibility
        """
        try:
            # Load full training data
            train_full = pd.read_csv(train_path)
            
            # Split into train/val with stratification
            self.train_data, self.val_data = train_test_split(
                train_full,
                train_size=train_size,
                test_size=1 - train_size,
                stratify=train_full["label"],
                random_state=random_state
            )
            
            # Apply preprocessing
            train_texts = self.preprocess_text(self.train_data["sentence"].copy())
            val_texts = self.preprocess_text(self.val_data["sentence"].copy())
            
            # Vectorize
            self.X_train, self.X_val = self.vectorize(train_texts, val_texts)
            
            # Extract labels
            self.Y_train = self.train_data["label"].values
            self.Y_val = self.val_data["label"].values
            
            # Get number of samples (works for both dense and sparse arrays)
            n_train = self.X_train.shape[0] if hasattr(self.X_train, 'shape') else len(self.X_train)
            n_val = self.X_val.shape[0] if hasattr(self.X_val, 'shape') else len(self.X_val)
            
            print(f"✓ Data loaded and preprocessed")
            print(f"  Train set: {n_train} samples")
            print(f"  Val set: {n_val} samples")
        except Exception as e:
            import traceback
            print(f"\n❌ Error during preprocessing:")
            traceback.print_exc()
            raise
    
    def train(self):
        """
        Train the classifier on the vectorized training data.
        """
        if self.X_train is None:
            raise RuntimeError("Data not loaded. Call load_and_preprocess_data() first.")
        
        try:
            self.model = self.get_model()
            self.model.fit(self.X_train, self.Y_train)
            print(f"✓ Model trained")
        except Exception as e:
            import traceback
            print(f"\n❌ Error during training:")
            traceback.print_exc()
            raise
    
    def evaluate(self) -> dict:
        """
        Evaluate the model on train and validation sets using MAE and accuracy.
        
        Returns:
            Dictionary with keys:
                - train_score: 1.0 - (MAE_train / 4.0)
                - train_mae: Mean absolute error on training set
                - train_accuracy: Exact match accuracy on training set
                - val_score: 1.0 - (MAE_val / 4.0)
                - val_mae: Mean absolute error on validation set
                - val_accuracy: Exact match accuracy on validation set
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        
        try:
            # Predictions
            Y_train_pred = self.model.predict(self.X_train)
            Y_val_pred = self.model.predict(self.X_val)
            
            # Training metrics
            mae_train = mean_absolute_error(self.Y_train, Y_train_pred)
            score_train = 1.0 - (mae_train / 4.0)
            accuracy_train = np.mean(self.Y_train == Y_train_pred)
            
            # Validation metrics
            mae_val = mean_absolute_error(self.Y_val, Y_val_pred)
            score_val = 1.0 - (mae_val / 4.0)
            accuracy_val = np.mean(self.Y_val == Y_val_pred)
            
            self.results = {
                "train_score": score_train,
                "train_mae": mae_train,
                "train_accuracy": accuracy_train,
                "val_score": score_val,
                "val_mae": mae_val,
                "val_accuracy": accuracy_val
            }
            
            print(f"✓ Evaluation complete")
            print(f"  Train - Score: {score_train:.4f}, MAE: {mae_train:.4f}, Accuracy: {accuracy_train:.4f}")
            print(f"  Val   - Score: {score_val:.4f}, MAE: {mae_val:.4f}, Accuracy: {accuracy_val:.4f}")
            
            return self.results
        except Exception as e:
            import traceback
            print(f"\n❌ Error during evaluation:")
            traceback.print_exc()
            raise
