# src/models/registry.py

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from xgboost import XGBClassifier


# ---------------------------
# Individual Model Builders
# ---------------------------
def get_logistic_regression():
    return LogisticRegression(
        max_iter=1000,
        class_weight='balanced'
    )


def get_random_forest():
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=42
    )


def get_gradient_boosting():
    return GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )


def get_xgboost():
    return XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        use_label_encoder=False,
        n_jobs=-1,
        random_state=42
    )


def get_ridge():
    return RidgeClassifier()
    

def get_svm():
    # Linear SVM (better for high-dimensional sparse data like TF-IDF)
    base_model = LinearSVC(class_weight='balanced')

    # Wrap with calibration for probability outputs
    return CalibratedClassifierCV(base_model, method="sigmoid")


# ---------------------------
# Model Registry
# ---------------------------
def get_models(selected_models=None):
    """
    Returns a dictionary of models.

    Args:
        selected_models (list or None):
            If provided, only returns those models.

    Available models:
        - logreg
        - random_forest
        - gradient_boosting
        - xgboost
        - ridge
        - svm
    """

    models = {
        "logreg": get_logistic_regression(),
        "random_forest": get_random_forest(),
        "gradient_boosting": get_gradient_boosting(),
        "xgboost": get_xgboost(),
        "ridge": get_ridge(),
        "svm": get_svm()
    }

    if selected_models is not None:
        models = {k: v for k, v in models.items() if k in selected_models}

    return models


"""
How to use it

from src.models.registry import get_models

models = get_models()

for name in models:
    print(name)
"""