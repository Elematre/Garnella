

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import LinearSVC
import xgboost as xgb

NUM_CLASSES = 5


# ============================================================
# Wrapper 1: Expected-value decoding for any probabilistic classifier
# ============================================================

class ExpectedValueClassifier(BaseEstimator, ClassifierMixin):
    """Wrap a classifier; replace argmax .predict() with E[y] = Σ i·p_i decoding.

    Rationale: for an ordinal target with MAE loss, argmax throws away
    information from the full probability distribution. Taking the expected
    value (then rounding) respects the ordinal structure and typically adds
    0.2–0.5 points on this task.

    Requires base_estimator.predict_proba. Labels must be 0..num_classes-1.
    """
    def __init__(self, base_estimator, num_classes: int = NUM_CLASSES):
        self.base_estimator = base_estimator
        self.num_classes = num_classes

    def fit(self, X, y):
        self.base_estimator_ = clone(self.base_estimator)
        self.base_estimator_.fit(X, y)
        self.classes_ = np.arange(self.num_classes)
        return self

    def predict(self, X):
        probs = self.base_estimator_.predict_proba(X)
        expected = (probs * np.arange(self.num_classes)).sum(axis=1)
        return np.clip(np.round(expected), 0, self.num_classes - 1).astype(int)

    def predict_proba(self, X):
        return self.base_estimator_.predict_proba(X)


# ============================================================
# Wrapper 2: Use a regressor as a classifier (round + clip)
# ============================================================

class RegressorAsClassifier(BaseEstimator, ClassifierMixin):
    """Train a regressor on integer labels; round + clip at predict time.

    Natural fit for MAE: the loss aligns directly, and we don't waste
    parameters modeling a categorical distribution over 5 classes when the
    underlying target is actually ordinal.
    """
    def __init__(self, base_regressor, num_classes: int = NUM_CLASSES):
        self.base_regressor = base_regressor
        self.num_classes = num_classes

    def fit(self, X, y):
        self.base_regressor_ = clone(self.base_regressor)
        self.base_regressor_.fit(X, np.asarray(y, dtype=float))
        self.classes_ = np.arange(self.num_classes)
        return self

    def predict(self, X):
        yhat = self.base_regressor_.predict(X)
        return np.clip(np.round(yhat), 0, self.num_classes - 1).astype(int)


# ============================================================
# Expected-value variants of your existing classifiers
# ============================================================

def get_logistic_regression_ev():
    return ExpectedValueClassifier(
        LogisticRegression(C=1.0, max_iter=1000)
    )

def get_mlp_ev():
    return ExpectedValueClassifier(
        MLPClassifier(
            hidden_layer_sizes=(128,),
            alpha=1e-2,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            max_iter=300,
            random_state=1,
        )
    )

def get_xgboost_ev():
    return ExpectedValueClassifier(
        xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            device="cuda", n_jobs=-1, random_state=1,
        )
    )

def get_linear_svm_ev():
    # LinearSVC has no predict_proba — calibrate it first.
    # cv=3 fits 3 SVCs for Platt scaling; slower but worth it for EV decoding.
    return ExpectedValueClassifier(
        CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=2000), cv=3)
    )


# ============================================================
# Regression-based models (direct MAE optimization)
# ============================================================

def get_ridge_regression():
    """Linear regression with L2 — the regression analogue of your LR baseline."""
    return RegressorAsClassifier(Ridge(alpha=1.0))

def get_mlp_regressor():
    """Same architecture as your MLP, but trained as a regressor."""
    return RegressorAsClassifier(
        MLPRegressor(
            hidden_layer_sizes=(128,),
            alpha=1e-2,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            max_iter=300,
            random_state=1,
        )
    )

# def get_xgboost_mae():
#     """XGBoost regressor that DIRECTLY optimizes MAE — aligns with the metric.

#     This is probably the single best Track 1 model you'll find. Unlike
#     classification XGBoost (which optimizes cross-entropy then argmax'es),
#     this optimizes L1 loss end-to-end.
#     """
#     return RegressorAsClassifier(
#         xgb.XGBRegressor(
#             n_estimators=500,
#             max_depth=6,
#             learning_rate=0.05,
#             subsample=0.9,
#             colsample_bytree=0.9,
#             objective="reg:absoluteerror",   # <- directly optimizes MAE
#             device="cuda",
#             n_jobs=-1,
#             random_state=1,
#         )
#     )

def get_xgboost_mae():
    return RegressorAsClassifier(
        xgb.XGBRegressor(
            n_estimators=1500,        # was 500
            max_depth=6,
            learning_rate=0.1,         # was 0.05
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:absoluteerror",
            device="cuda",
            random_state=1,
        )
    )