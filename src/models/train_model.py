from sklearn.metrics import accuracy_score, log_loss

def train_and_evaluate(models, X_train, y_train, X_cv, y_cv, X_test=None, y_test=None):

    results = {}

    for name, model in models.items():
        print(f"Training: {name}")

        model.fit(X_train, y_train)

        # Train metrics
        train_acc = accuracy_score(y_train, model.predict(X_train))

        # CV metrics
        cv_acc = accuracy_score(y_cv, model.predict(X_cv))

        # Test metrics
        test_acc = accuracy_score(y_test, model.predict(X_test)) if X_test is not None and y_test is not None else None

        try:
            train_loss = log_loss(y_train, model.predict_proba(X_train))
            cv_loss = log_loss(y_cv, model.predict_proba(X_cv))
            test_loss = log_loss(y_test, model.predict_proba(X_test)) if X_test is not None and y_test is not None else None
        except:
            train_loss = None
            cv_loss = None
            test_loss = None

        results[name] = {
            "train_accuracy": train_acc,
            "train_loss": train_loss,
            "cv_accuracy": cv_acc,
            "cv_loss": cv_loss,
            "test_accuracy": test_acc,
            "test_loss": test_loss
        }

    return results

"""
# How to use it

from src.data.loader import load_training_data
from src.feature.preprocessing import preprocess_pipeline
from src.models.registry import get_models

# 1. Load Data
df = load_training_data()

# 2. Preprocess and Split
X_train, X_cv, X_test, y_train, y_cv, y_test, vectorizers = preprocess_pipeline(df)

# 3. Get Models from Registry
models = get_models()

# 4. Train and Evaluate
results = train_and_evaluate(models, X_train, y_train, X_cv, y_cv, X_test, y_test)

# 5. Print Results
for model_name, metrics in results.items():
    print(f"Model: {model_name}")
    print(f"  Train Accuracy: {metrics['train_accuracy']:.4f}")
    print(f"  Train Loss: {metrics['train_loss']}")
    print(f"  CV Accuracy: {metrics['cv_accuracy']:.4f}")
    print(f"  CV Loss: {metrics['cv_loss']}")
    print(f"  Test Accuracy: {metrics['test_accuracy']:.4f}" if metrics['test_accuracy'] else "  Test Accuracy: None")
    print(f"  Test Loss: {metrics['test_loss']}")
"""