import numpy as np
from scipy.sparse import issparse


# ---------------------------------------------------------------------------
# Global Feature Importance
# ---------------------------------------------------------------------------

def get_feature_importance(model, feature_names=None, top_k=20):
    """
    Returns top-k important features. Unwraps CalibratedClassifierCV automatically.
    Works for: Logistic Regression, SVM, Random Forest, XGBoost, etc.
    """
    # Unwrap CalibratedClassifierCV — cv=3 stores a list of calibrated_classifiers_
    base = model
    if hasattr(model, 'calibrated_classifiers_'):
        # Each calibrated classifier has an .estimator attribute
        base = model.calibrated_classifiers_[0].estimator
    elif hasattr(model, 'estimator'):
        base = model.estimator

    if hasattr(base, "coef_"):
        importance = np.abs(base.coef_).mean(axis=0)
    elif hasattr(base, "feature_importances_"):
        importance = base.feature_importances_
    else:
        raise ValueError(
            f"Model '{type(base).__name__}' does not support feature importance. "
            "Needs .coef_ or .feature_importances_"
        )

    indices = np.argsort(importance)[::-1][:top_k]
    results = []
    for idx in indices:
        feature = feature_names[idx] if feature_names is not None else f"feature_{idx}"
        results.append({"feature": str(feature), "importance": float(importance[idx])})
    return results


# ---------------------------------------------------------------------------
# LIME — Local Explanation (Tabular)
# ---------------------------------------------------------------------------

def explain_with_lime(model, X_train, X_sample, feature_names=None,
                      num_features=10, num_samples=500):
    """
    Generates a LIME explanation for a single sample.

    Parameters
    ----------
    model         : fitted sklearn estimator with predict_proba
    X_train       : training data (sparse or dense) — used as background
    X_sample      : single sample, shape (1, n_features), sparse or dense
    feature_names : list of feature name strings (optional)
    num_features  : top features to show per class
    num_samples   : perturbations LIME generates internally

    Returns
    -------
    dict:
        predicted_class : int
        class_names     : list[str]
        explanations    : list[{feature, weight}]  for the predicted class
        all_classes     : list[{class_index, class_name, features}]
    """
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError:
        raise ImportError("Install lime: pip install lime")

    # ── Step 1: Random subset of 100 rows as background ─────────────────────
    # Full X_train can be 3000+ rows x 55000 cols → ~1.3 GB dense
    # 100 rows is enough for LIME to learn the feature value distribution
    n_background = min(100, X_train.shape[0])
    rng = np.random.RandomState(42)
    bg_idx = rng.choice(X_train.shape[0], n_background, replace=False)
    X_background = X_train[bg_idx]

    # ── Step 2: Convert sparse → dense (LIME requires dense numpy arrays) ───
    X_bg_dense = X_background.toarray() if issparse(X_background) else np.asarray(X_background)
    X_sample_dense = X_sample.toarray()[0] if issparse(X_sample) else np.asarray(X_sample).flatten()

    # ── Step 3: Class names ──────────────────────────────────────────────────
    n_classes = len(model.classes_) if hasattr(model, "classes_") else 9
    class_labels = list(model.classes_) if hasattr(model, "classes_") else list(range(1, n_classes + 1))
    class_names = [f"Class {c}" for c in class_labels]

    # ── Step 4: Build LimeTabularExplainer with the background subset ────────
    explainer = LimeTabularExplainer(
        training_data=X_bg_dense,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
        discretize_continuous=True,
        random_state=42
    )

    # ── Step 5: Explain the input sample ────────────────────────────────────
    # LIME perturbs X_sample_dense ~num_samples times, calls model.predict_proba
    # on each, then fits a local linear model to approximate the boundary
    exp = explainer.explain_instance(
        data_row=X_sample_dense,
        predict_fn=model.predict_proba,
        num_features=num_features,
        num_samples=num_samples,
        top_labels=n_classes
    )

    # ── Step 6: Predicted class ──────────────────────────────────────────────
    predicted_class = int(model.predict(X_sample_dense.reshape(1, -1))[0])

    # ── Step 7: Build JSON-serializable output ───────────────────────────────
    # LIME labels are 0-indexed; class_labels are 1-indexed (Class 1..9)
    all_classes = []
    for label in sorted(exp.available_labels()):
        features_list = [
            {"feature": str(feat), "weight": float(w)}
            for feat, w in exp.as_list(label=label)
        ]
        cls_index = int(class_labels[label]) if label < len(class_labels) else int(label)
        cls_name = class_names[label] if label < len(class_names) else f"Class {label}"
        all_classes.append({
            "class_index": cls_index,
            "class_name": cls_name,
            "features": features_list
        })

    # Explanation for the predicted class specifically
    pred_label = predicted_class - 1 if predicted_class >= 1 else 0
    pred_label = max(0, min(pred_label, n_classes - 1))
    explanations = [
        {"feature": str(feat), "weight": float(w)}
        for feat, w in exp.as_list(label=pred_label)
    ]

    return {
        "predicted_class": predicted_class,
        "class_names": class_names,
        "explanations": explanations,
        "all_classes": all_classes
    }


# ---------------------------------------------------------------------------
# SHAP — Local + Global Explanation
# ---------------------------------------------------------------------------

def explain_with_shap(model, X_train, X_sample, feature_names=None,
                      top_k=15, background_samples=50):
    """
    Computes SHAP values for a single sample using KernelExplainer.
    Uses the top-500 variance trick from the reference project to avoid
    memory crashes on high-dimensional sparse TF-IDF matrices.

    Parameters
    ----------
    model              : fitted sklearn estimator with predict_proba
    X_train            : training data (sparse or dense)
    X_sample           : single sample, shape (1, n_features)
    feature_names      : list of feature name strings (optional)
    top_k              : top features to return per class
    background_samples : rows to use for KernelExplainer background

    Returns
    -------
    dict:
        predicted_class : int
        expected_value  : list[float]  base values per class
        shap_values     : list[{class_index, class_name, features:[{feature, shap_value}]}]
        top_features    : list[{feature, mean_abs_shap}]  global importance
    """
    try:
        from shap import KernelExplainer, kmeans as shap_kmeans
    except ImportError:
        raise ImportError("Install shap: pip install shap")

    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # ── Step 1: Convert sparse → dense ──────────────────────────────────────
    X_train_dense = X_train.toarray() if issparse(X_train) else np.asarray(X_train, dtype=np.float64)
    X_sample_dense = X_sample.toarray() if issparse(X_sample) else np.asarray(X_sample, dtype=np.float64)

    n_total_features = X_train_dense.shape[1]
    n_classes = len(model.classes_) if hasattr(model, "classes_") else 9
    class_labels = list(model.classes_) if hasattr(model, "classes_") else list(range(1, n_classes + 1))
    class_names = [f"Class {c}" for c in class_labels]

    # ── Step 2: Top-500 features by variance (memory safety trick) ──────────
    # Reference project uses this same trick to avoid 2 GB RAM allocation.
    # We compute variance across the background rows and keep only the
    # top-500 most variable features — these carry the most signal anyway.
    actual_bg = min(background_samples, X_train_dense.shape[0])
    rng = np.random.RandomState(42)
    bg_idx = rng.choice(X_train_dense.shape[0], actual_bg, replace=False)
    bg_dense = X_train_dense[bg_idx]

    variances = bg_dense.var(axis=0)
    top500_idx = np.argsort(-variances)[:500]
    bg_small = bg_dense[:, top500_idx]           # shape (actual_bg, 500)

    # ── Step 3: Wrapper that pads 500-feature slice back to full size ────────
    # KernelExplainer will call this with (n, 500) arrays.
    # We pad back to (n, n_total_features) before calling predict_proba.
    def shap_predict(X_small):
        X_full = np.zeros((X_small.shape[0], n_total_features), dtype=np.float64)
        X_full[:, top500_idx] = X_small
        return model.predict_proba(X_full)

    # ── Step 4: Build KernelExplainer on the small background ───────────────
    explainer = KernelExplainer(shap_predict, bg_small[:2])

    # ── Step 5: Compute SHAP values for the input sample ────────────────────
    X_sample_small = X_sample_dense[:, top500_idx]   # slice to 500 features
    shap_vals = explainer.shap_values(X_sample_small, nsamples=50, silent=True)
    # shap_vals: list of arrays, one per class, each shape (1, 500)

    # ── Step 6: Predicted class ──────────────────────────────────────────────
    predicted_class = int(model.predict(X_sample_dense)[0])

    # ── Step 7: Build per-class SHAP output ─────────────────────────────────
    shap_per_class = []
    all_abs_shap = np.zeros(500)

    for i, sv in enumerate(shap_vals):
        sv_row = sv[0] if sv.ndim == 2 else sv     # shape (500,)
        all_abs_shap += np.abs(sv_row)

        top_idx = np.argsort(-np.abs(sv_row))[:top_k]
        features_list = []
        for idx in top_idx:
            # Map back from 500-feature index to original feature name
            original_idx = top500_idx[idx]
            fname = feature_names[original_idx] if (
                feature_names is not None and original_idx < len(feature_names)
            ) else f"feature_{original_idx}"
            features_list.append({
                "feature": str(fname),
                "shap_value": float(sv_row[idx])
            })

        shap_per_class.append({
            "class_index": int(class_labels[i]),
            "class_name": class_names[i],
            "features": features_list
        })

    # ── Step 8: Global top features (mean |SHAP| across all classes) ─────────
    mean_abs = all_abs_shap / len(shap_vals)
    global_top_idx = np.argsort(-mean_abs)[:top_k]
    top_features = []
    for idx in global_top_idx:
        original_idx = top500_idx[idx]
        fname = feature_names[original_idx] if (
            feature_names is not None and original_idx < len(feature_names)
        ) else f"feature_{original_idx}"
        top_features.append({
            "feature": str(fname),
            "mean_abs_shap": float(mean_abs[idx])
        })

    # ── Step 9: Expected values ──────────────────────────────────────────────
    ev = explainer.expected_value
    expected_value = ev.tolist() if hasattr(ev, "tolist") else list(ev)

    return {
        "predicted_class": predicted_class,
        "expected_value": expected_value,
        "shap_values": shap_per_class,
        "top_features": top_features
    }
