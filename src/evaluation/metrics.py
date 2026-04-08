# src/evaluation/metrics.py

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)


# ---------------------------
# Core Evaluation
# ---------------------------
def evaluate_classification(model, X, y):
    """
    Evaluates a classification model.

    Returns:
        dict of metrics
    """

    results = {}

    # Predictions
    y_pred = model.predict(X)

    # Basic metrics
    results["accuracy"] = accuracy_score(y, y_pred)
    results["f1_macro"] = f1_score(y, y_pred, average="macro")
    results["precision_macro"] = precision_score(y, y_pred, average="macro", zero_division=0)
    results["recall_macro"] = recall_score(y, y_pred, average="macro", zero_division=0)

    # Log Loss (if available)
    try:
        y_prob = model.predict_proba(X)
        results["log_loss"] = log_loss(y, y_prob)
    except:
        results["log_loss"] = None

    return results


# ---------------------------
# Confusion Matrix
# ---------------------------
def get_confusion_matrix(model, X, y):
    """
    Returns confusion matrix
    """

    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)

    return cm


# ---------------------------
# Per-Class Accuracy
# ---------------------------
def per_class_accuracy(model, X, y):
    """
    Computes accuracy per class
    """

    y_pred = model.predict(X)

    results = {}

    for cls in np.unique(y):
        idx = (y == cls)

        if idx.sum() == 0:
            continue

        acc = accuracy_score(y[idx], y_pred[idx])
        results[int(cls)] = acc

    return results


# ---------------------------
# Aggregate Evaluation (Train + Val/Test)
# ---------------------------
def evaluate_all(model, X_train, y_train, X_val, y_val):
    """
    Returns evaluation on both train and validation sets
    """

    train_metrics = evaluate_classification(model, X_train, y_train)
    val_metrics = evaluate_classification(model, X_val, y_val)

    return {
        "train": train_metrics,
        "validation": val_metrics
    }


# ---------------------------
# Pretty Print (Optional)
# ---------------------------
def print_metrics(metrics_dict, model_name="model"):
    """
    Nicely prints metrics
    """

    print(f"\n📊 Results for {model_name}")

    for split, metrics in metrics_dict.items():
        print(f"\n--- {split.upper()} ---")
        for key, value in metrics.items():
            if value is not None:
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: None")


"""
How to use it 

from src.evaluation.metrics import evaluate_all, print_metrics
from src.models.registry import get_models
from src.data.loader import load_training_data
from src.feature.preprocessing import preprocess_pipeline

# 1. Load and Preprocess
df = load_training_data()
X_train, X_cv, X_test, y_train, y_cv, y_test, vectorizers = preprocess_pipeline(df)

# 2. Train a model
models = get_models(selected_models=["logistic_regression"])
lr = models["logistic_regression"]
lr.fit(X_train, y_train)

# 3. Evaluate
results = evaluate_all(lr, X_train, y_train, X_cv, y_cv)

# 4. Print Results
print_metrics(results, model_name="Logistic Regression")
"""