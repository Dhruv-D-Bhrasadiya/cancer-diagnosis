import pandas as pd
from sklearn.metrics import accuracy_score


# Group-wise Accuracy
def group_accuracy(model, X, y, group_values):
    """
    Computes accuracy per group.

    Args:
        model: trained model
        X: feature matrix
        y: true labels
        group_values: list/array (e.g., df['Gene'])

    Returns:
        DataFrame with accuracy per group
    """

    preds = model.predict(X)

    df = pd.DataFrame({
        "group": group_values,
        "true": y,
        "pred": preds
    })

    results = (
        df.groupby("group")
        .apply(lambda x: accuracy_score(x["true"], x["pred"]))
        .reset_index(name="accuracy")
        .sort_values(by="accuracy", ascending=False)
    )

    return results


# Group-wise Log Loss
def group_log_loss(model, X, y, group_values):
    """
    Computes log loss per group (if model supports predict_proba)
    """

    try:
        probs = model.predict_proba(X)
    except:
        raise ValueError("Model does not support probability predictions")

    df = pd.DataFrame({
        "group": group_values,
        "true": y
    })

    results = []

    for group in df["group"].unique():
        idx = df["group"] == group

        if idx.sum() < 2:
            continue  # skip tiny groups

        from sklearn.metrics import log_loss

        loss = log_loss(y[idx], probs[idx])

        results.append({
            "group": group,
            "log_loss": loss,
            "count": idx.sum()
        })

    return pd.DataFrame(results).sort_values(by="log_loss")


# Fairness Gap
def fairness_gap(group_metric_df, metric_col="accuracy"):
    """
    Computes disparity between best and worst groups
    """

    max_val = group_metric_df[metric_col].max()
    min_val = group_metric_df[metric_col].min()

    return {
        "max": max_val,
        "min": min_val,
        "gap": max_val - min_val
    }


# Full Fairness Report
def fairness_report(model, X, y, group_values):
    """
    Returns full fairness analysis
    """

    acc_df = group_accuracy(model, X, y, group_values)
    acc_gap = fairness_gap(acc_df, "accuracy")

    report = {
        "group_accuracy": acc_df,
        "accuracy_gap": acc_gap
    }

    # Try log loss if available
    try:
        loss_df = group_log_loss(model, X, y, group_values)
        loss_gap = fairness_gap(loss_df, "log_loss")

        report["group_log_loss"] = loss_df
        report["log_loss_gap"] = loss_gap

    except:
        report["group_log_loss"] = None
        report["log_loss_gap"] = None

    return report


"""
# How to use it

from src.evaluation.fairness import fairness_report
from src.models.registry import get_models
from src.data.loader import load_training_data
from src.feature.preprocessing import preprocess_pipeline

# 1. Load and Preprocess
df = load_training_data()
X_train, X_cv, X_test, y_train, y_cv, y_test, vectorizers = preprocess_pipeline(df)

# 2. Train a model
models = get_models(selected_models=["random_forest"])
rf = models["random_forest"]
rf.fit(X_train, y_train)

# 3. Run Fairness Report (using 'Gene' as the sensitive/grouping attribute)
# Note: We need the original group values corresponding to the CV/Test set
# In this pipeline, we'd extract them from the split dataframes
report = fairness_report(rf, X_cv, y_cv, group_values=df.loc[X_cv.index, 'Gene'])

# 4. Print results
print("Accuracy Gap:", report["accuracy_gap"])
print(report["group_accuracy"].head())

if report["group_log_loss"] is not None:
    print("Log Loss Gap:", report["log_loss_gap"])
    print(report["group_log_loss"].head())
"""