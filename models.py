from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import xgboost as xgb


####### BASELINES ########

def get_logistic_regression():
    return LogisticRegression(C=1.0, max_iter=1000, n_jobs=-1)


def get_linear_svm():
    return LinearSVC(C=1.0, max_iter=2000)


def get_knn():
    return KNeighborsClassifier(n_neighbors=15, n_jobs=-1)


def get_mlp():
    # fixes the 98%/54% overfit: smaller net, more L2, early stopping
    return MLPClassifier(
        hidden_layer_sizes=(128,),
        alpha=1e-2,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        max_iter=300,
        random_state=1,
    )


def get_random_forest():
    return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=1)

def get_random_forest_v2():
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=10,        # limit tree depth
        min_samples_leaf=5,  # require at least 5 samples per leaf
        n_jobs=-1,
        random_state=1,
    )
def get_xgboost():
    return xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        device="cuda", n_jobs=-1, random_state=1,
    )


####### TUNED VARIANTS ########

def get_logistic_regression_tuned():
    param_grid = {"C": [0.01, 0.1, 1.0, 10.0]}
    return GridSearchCV(
        LogisticRegression(max_iter=2000, n_jobs=-1),
        param_grid, cv=3, scoring="neg_mean_absolute_error", n_jobs=-1,
    )


def get_xgboost_tuned():
    param_grid = {
        "max_depth": [3, 4, 6],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
    }
    return GridSearchCV(
        xgb.XGBClassifier(n_estimators=300, device="cuda", n_jobs=-1, random_state=1),
        param_grid, cv=3, scoring="neg_mean_absolute_error", n_jobs=1,
    )