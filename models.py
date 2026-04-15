from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

####### BASELINES ########

def get_logistic_regression():
    return LogisticRegression(C=1.0, max_iter=100)

def get_linear_svm():
    return LinearSVC(C=1.0, max_iter=1000)

# NOTE: not sure if needed? would need low-dimensional dense embeddings first
# def get_random_forest():
#     # some start parameters, so it doesn't take too long to train, can be tuned later
#     return RandomForestClassifier(n_estimators=100, n_jobs=-1)

# needed? 
def get_mlp():
    # some start parameters, can be tuned later
    return MLPClassifier(random_state=1,max_iter=300)

# 1D CNN
