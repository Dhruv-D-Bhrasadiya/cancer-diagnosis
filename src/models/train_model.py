from sklearn.metrics import accuracy_score, log_loss

def train_and_evaluate(models, X_train, y_train, X_val, y_val):

    results = {}

    for name, model in models.items():
        print(f"Training: {name}")

        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)

        try:
            y_prob = model.predict_proba(X_val)
            loss = log_loss(y_val, y_prob)
        except:
            loss = None

        results[name] = {
            "accuracy": acc,
            "log_loss": loss
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
results = train_and_evaluate(models, X_train, y_train, X_cv, y_cv)

# 5. Print Results
for model_name, metrics in results.items():
    print(f"Model: {model_name}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Log Loss: {metrics['log_loss']}")
"""