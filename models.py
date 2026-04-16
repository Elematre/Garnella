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

# # needed? 
# def get_mlp():
#     # some start parameters, can be tuned later
#     return MLPClassifier(random_state=1,max_iter=300)

# 1D CNN




# more stuff
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb


def get_mlp():
    return MLPClassifier(hidden_layer_sizes=(256,), max_iter=300, random_state=1)

def get_random_forest():
    return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=1)

def get_xgboost():
    return xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        tree_method="gpu_hist", n_jobs=-1, random_state=1
    )

def get_knn():
    return KNeighborsClassifier(n_neighbors=15, n_jobs=-1)