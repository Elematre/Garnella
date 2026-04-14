from sklearn.metrics import mean_absolute_error
import numpy as np

# for structure Output = Classifier(Embeddings(x))
def train_loop(train_data, test_data, combinations):
    """
    combinations: tuples of (vectorizer_func, model_func) tuples
    eg (get_bagOfWords_embeddings, get_logistic_regression)
    """
    results = []

    for vectorizer_func, model_func in combinations:
        # 1) Vectorize -> get embeddings
        ## TODO: add other workflow if we want to use pretrained word embeddings, these don't take same arguments
        X_train, X_val = vectorizer_func(train_data["sentence"], test_data["sentence"])
        # extract ground truth labels
        Y_train, Y_val = train_data["label"], test_data["label"]

        # 2) Train
        # instantiate model, so it is a clean state
        model = model_func()
        model.fit(X_train, Y_train)

        # 3) Evaluate
        Y_train_pred = model.predict(X_train)
        Y_val_pred = model.predict(X_val)
        score_train, mae_train, accuracy_train, score_val, mae_val, accuracy_val = evaluateMAE(Y_train_pred, Y_train, Y_val_pred, Y_val)

        print(f"Combination: {vectorizer_func.__name__} + {type(model).__name__}")
        print(f"Training Score: {score_train:.4f}, MAE: {mae_train:.4f}, Accuracy: {accuracy_train:.4f}")
        print(f"Validation Score: {score_val:.4f}, MAE: {mae_val:.4f}, Accuracy: {accuracy_val:.4f}")
        
        results.append({
            "vectorizer": vectorizer_func.__name__,
            "model": type(model).__name__,
            "training_score": score_train,
            "training_mae": mae_train,
            "training_accuracy": accuracy_train,
            "validation_score": score_val,
            "validation_mae": mae_val,
            "validation_accuracy": accuracy_val
        })

    return results

def evaluateMAE(Y_train_pred, Y_train, Y_val_pred, Y_val): 
    # score on training set
    mae_train = mean_absolute_error(Y_train, Y_train_pred)
    score_train = 1.0 - (mae_train / 4.0)
    accuracy_train = np.mean(Y_train == Y_train_pred)

    # score on validation set
    mae_val = mean_absolute_error(Y_val, Y_val_pred)
    score_val = 1.0 - (mae_val / 4.0)
    accuracy_val = np.mean(Y_val == Y_val_pred)
    
    return score_train, mae_train, accuracy_train, score_val, mae_val, accuracy_val