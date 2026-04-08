import numpy as np


# Feature Importance (Global)
def get_feature_importance(model, feature_names=None, top_k=20):
    """
    Returns top-k important features (if model supports it)

    Works for:
    - Logistic Regression
    - Linear models
    - Tree-based models
    """

    if hasattr(model, "coef_"):
        importance = np.abs(model.coef_).mean(axis=0)

    elif hasattr(model, "feature_importances_"):
        importance = model.feature_importances_

    else:
        raise ValueError("Model does not support feature importance")

    indices = np.argsort(importance)[::-1][:top_k]

    results = []

    for idx in indices:
        feature = feature_names[idx] if feature_names is not None else idx
        results.append({
            "feature": feature,
            "importance": importance[idx]
        })

    return results


# SHAP (Global + Local)
def compute_shap_values(model, X_sample):
    """
    Computes SHAP values for a sample of data
    """

    try:
        import shap
    except ImportError:
        raise ImportError("Install shap: pip install shap")

    explainer = shap.Explainer(model, X_sample)
    shap_values = explainer(X_sample)

    return shap_values


# SHAP Summary Plot
def plot_shap_summary(shap_values):
    """
    Displays SHAP summary plot
    """

    import shap
    shap.plots.beeswarm(shap_values)


# SHAP Force Plot (Local Explanation)
def plot_shap_force(shap_values, index=0):
    """
    Explains a single prediction
    """

    import shap
    shap.plots.force(shap_values[index])


# LIME (Local Explanation)
def explain_with_lime(model, X_train, X_sample, feature_names=None, class_names=None):
    """
    Generates LIME explanation for a single sample
    """

    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError:
        raise ImportError("Install lime: pip install lime")

    explainer = LimeTabularExplainer(
        X_train.toarray(),
        feature_names=feature_names,
        class_names=class_names,
        mode="classification"
    )

    exp = explainer.explain_instance(
        X_sample.toarray()[0],
        model.predict_proba,
        num_features=10
    )

    return exp

"""
How to use it 

from src.evaluation.interpretability import get_feature_importance, explain_with_lime
from src.models.registry import get_models
from src.data.loader import load_training_data
from src.feature.preprocessing import preprocess_pipeline

# 1. Load and Preprocess
df = load_training_data()
X_train, X_cv, X_test, y_train, y_cv, y_test, vectorizers = preprocess_pipeline(df)

# Get feature names from vectorizers for better interpretability
# (Assuming vectorizers is a list of fitted Tfidf/Count vectorizers)
feature_names = []
for v in vectorizers:
    feature_names.extend(v.get_feature_names_out())

# 2. Train a model
models = get_models(selected_models=["logistic_regression"])
lr = models["logistic_regression"]
lr.fit(X_train, y_train)

# 3. Global Interpretability: Feature Importance
importance = get_feature_importance(lr, feature_names=feature_names, top_k=10)
print("Top Features:", importance)

# 4. Local Interpretability: LIME
# Explain the first instance of the test set
exp = explain_with_lime(
    model=lr,
    X_train=X_train,
    X_sample=X_test[0],
    feature_names=feature_names,
    class_names=[str(i) for i in range(9)] # Assuming 9 classes for cancer types
)
exp.show_in_notebook()
"""