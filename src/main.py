from data.loader import load_all_data
from feature.preprocessing import preprocess_pipeline
from models.registry import get_models

from evaluation.metrics import evaluate_all, get_confusion_matrix, per_class_accuracy, print_metrics
from evaluation.fairness import fairness_report
from utils.logger import TBLogger
from utils.tracking import CarbonTracker
import os
import joblib
from datetime import datetime

from tqdm import tqdm
import time


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

    # 4. Logger
    logger = TBLogger(experiment_name="multi_model_experiment")

    results = {}

    # 5. Training Loop
    pbar = tqdm(models.items(), desc="Overall Progress")
    for step, (name, model) in enumerate(pbar):
        pbar.set_postfix(model=name)
        print(f"\n[INFO] Training: {name}")
        start_time = time.time()

        # Carbon Tracker per model
        tracker = CarbonTracker(experiment_name=f"model_{name}")
        tracker.start()

        # Train
        model.fit(X_train, y_train)
        
        # Stop Carbon Tracker
        emissions = tracker.stop()
        print(f"\n[INFO] Carbon Emissions for {name}: {emissions:.6f} kg CO2")

        # Evaluate all splits
        all_metrics = evaluate_all(model, X_train, y_train, X_cv, y_cv, X_test, y_test)
        print_metrics(all_metrics, model_name=name)

        # Confusion Matrix
        cm = get_confusion_matrix(model, X_cv, y_cv)
        print(f"\n[INFO] Confusion Matrix for {name} (Validation):\n{cm}")

        # Per-class accuracy
        pca = per_class_accuracy(model, X_cv, y_cv)
        print(f"\n[INFO] Per-Class Accuracy for {name} (Validation):\n{pca}")

        # Log to TensorBoard
        # Log Carbon Emissions
        logger.log_metrics(name, {"emissions": emissions}, split="CARBON", step=0)
        
        # Log metrics for train, validation, and test
        for split, split_metrics in all_metrics.items():
            logger.log_metrics(name, split_metrics, split=split.upper(), step=0)

        # Set validation metrics for downstream tasks
        metrics = all_metrics["validation"]

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
        
        elapsed_time = time.time() - start_time
        print(f"[INFO] Finished {name} in {elapsed_time:.2f} seconds.")

    logger.close()

    # 7. Final Results
    print("\n[INFO] Final Results:")
    for model_name, metrics in results.items():
        print(f"{model_name}: {metrics}")


if __name__ == "__main__":
    main()