# train_loop_caching.py
from sklearn.metrics import mean_absolute_error
import numpy as np

def train_loop(train_data, test_data, combinations):
    results = []
    embedding_cache = {}

    for vectorizer_func, model_func in combinations:
        func_name = getattr(vectorizer_func, '__name__', None) or getattr(vectorizer_func, 'func', vectorizer_func).__name__

        if func_name not in embedding_cache:
            X_train, X_val = vectorizer_func(train_data["sentence"], test_data["sentence"])
            embedding_cache[func_name] = (X_train, X_val)
            print(f"Done with embeddings: {func_name}")
        else:
            X_train, X_val = embedding_cache[func_name]
            print(f"Using cached embeddings: {func_name}")

        Y_train, Y_val = train_data["label"], test_data["label"]

        model = model_func()
        model.fit(X_train, Y_train)

        Y_train_pred = model.predict(X_train)
        Y_val_pred = model.predict(X_val)
        score_train, mae_train, accuracy_train, score_val, mae_val, accuracy_val = evaluateMAE(Y_train_pred, Y_train, Y_val_pred, Y_val)

        
        print(f"Combination: {func_name} + {model_func.__name__}")
        print(f"Training Score: {score_train:.4f}, MAE: {mae_train:.4f}, Accuracy: {accuracy_train:.4f}")
        print(f"Validation Score: {score_val:.4f}, MAE: {mae_val:.4f}, Accuracy: {accuracy_val:.4f}")

        results.append({
            "vectorizer": func_name,
            "model": model_func.__name__,
            "training_score": score_train,
            "training_mae": mae_train,
            "training_accuracy": accuracy_train,
            "validation_score": score_val,
            "validation_mae": mae_val,
            "validation_accuracy": accuracy_val
        })

    return results

def evaluateMAE(Y_train_pred, Y_train, Y_val_pred, Y_val):
    mae_train = mean_absolute_error(Y_train, Y_train_pred)
    score_train = 1.0 - (mae_train / 4.0)
    accuracy_train = np.mean(Y_train == Y_train_pred)

    mae_val = mean_absolute_error(Y_val, Y_val_pred)
    score_val = 1.0 - (mae_val / 4.0)
    accuracy_val = np.mean(Y_val == Y_val_pred)

    return score_train, mae_train, accuracy_train, score_val, mae_val, accuracy_val