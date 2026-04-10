from evaluation.metrics import evaluate_all, get_confusion_matrix, per_class_accuracy, print_metrics

def train_and_evaluate(models, X_train, y_train, X_cv, y_cv, X_test=None, y_test=None):

    results = {}

    for name, model in models.items():
        print(f"Training: {name}")

        model.fit(X_train, y_train)

        all_metrics = evaluate_all(model, X_train, y_train, X_cv, y_cv, X_test, y_test)
        
        # Pretty Print Results
        print_metrics(all_metrics, model_name=name)

        # Additional metrics
        cm = get_confusion_matrix(model, X_cv, y_cv)
        pca = per_class_accuracy(model, X_cv, y_cv)
        
        results[name] = {
            "metrics": all_metrics,
            "confusion_matrix": cm,
            "per_class_accuracy": pca
        }

    return results

"""
# How to use it

from src.data.loader import load_training_data
from src.feature.preprocessing import preprocess_pipeline
from src.models.registry import get_models
from src.models.train_model import train_and_evaluate

# 1. Load Data
df = load_training_data()

# 2. Preprocess and Split
X_train, X_cv, X_test, y_train, y_cv, y_test, vectorizers, _ = preprocess_pipeline(df)

# 3. Get Models from Registry
models = get_models()

# 4. Train and Evaluate
results = train_and_evaluate(models, X_train, y_train, X_cv, y_cv, X_test, y_test)
"""
