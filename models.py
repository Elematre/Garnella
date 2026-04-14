from sklearn.linear_model import LogisticRegression

def get_logistic_regression():
    return LogisticRegression(C=1.0, max_iter=100)