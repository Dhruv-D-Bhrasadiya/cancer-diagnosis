from data.loader import load_all_data
from feature.preprocessing import preprocess_pipeline
from models.registry import get_models

from evaluation.metrics import evaluate_classification
from evaluation.fairness import fairness_report
from utils.logger import TBLogger
from utils.tracking import CarbonTracker
import os
import joblib
from datetime import datetime


def main():

    print("[INFO] Starting pipeline...\n")

    # 1. Load Data
    print("[INFO] Loading data...")
    train_df, test_df = load_all_data()

    # 2. Preprocessing
    print("[INFO] Preprocessing...")
    X_train, X_cv, X_test, y_train, y_cv, y_test, vectorizers, cv_df = preprocess_pipeline(train_df)

    # 3. Models
    print("[INFO] Loading models...")
    models = get_models()

    # 4. Logger + Carbon Tracker
    logger = TBLogger(experiment_name="multi_model_experiment")
    tracker = CarbonTracker(experiment_name="multi_model_experiment")

    tracker.start()

    results = {}

    # 5. Training Loop
    for step, (name, model) in enumerate(models.items()):
        print(f"\n[INFO] Training: {name}")

        # Train
        model.fit(X_train, y_train)

        # Evaluate
        metrics = evaluate_classification(model, X_cv, y_cv)

        print(f"Metrics: {metrics}")

        # Log to TensorBoard
        logger.log_metrics(name, metrics, step=step)

        # Fairness (using Gene as group)
        try:
            fairness = fairness_report(
                model,
                X_cv,
                y_cv,
                group_values=cv_df['Gene']
            )

            print("Fairness Gap:", fairness["accuracy_gap"])

        except Exception as e:
            print("Fairness skipped:", e)

        results[name] = metrics

        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{name}_{timestamp}.joblib"
        model_path = os.path.join("outputs", "models", model_filename)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        print(f"[INFO] Saved model to {model_path}")

    # 6. Stop Carbon Tracking
    emissions = tracker.stop()
    print(f"\n[INFO] Carbon Emissions: {emissions:.6f} kg CO2")

    logger.log_metrics("carbon", {"emissions": emissions})

    logger.close()

    # 7. Final Results
    print("\n[INFO] Final Results:")
    for model_name, metrics in results.items():
        print(f"{model_name}: {metrics}")


if __name__ == "__main__":
    main()