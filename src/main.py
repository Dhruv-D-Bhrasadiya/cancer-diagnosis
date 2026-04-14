from data.loader import load_processed_data, load_all_data
from feature.preprocessing import preprocess_pipeline, get_all_preprocessing_configs
from models.registry import get_models

from evaluation.metrics import evaluate_all, get_confusion_matrix, per_class_accuracy, print_metrics
from evaluation.fairness import fairness_report
from utils.logger import TBLogger
from utils.tracking import CarbonTracker
import os
import joblib
from datetime import datetime
from pathlib import Path

from tqdm import tqdm
import time
import json
from itertools import count


def main(use_existing_data=True, sample_configs=None):
    """
    Main training pipeline that iterates through preprocessing configurations.
    
    Args:
        use_existing_data: If True, load from data/processed. If False, regenerate.
        sample_configs: If provided, only train on these configs (for quick testing).
                       Otherwise, all configs are used.
    """
    
    print("[INFO] Starting pipeline...\n")
    
    # 1. Load or Generate Data
    print("[INFO] Loading data...")
    try:
        if use_existing_data:
            print("[INFO] Loading pre-processed data from data/processed/...")
            train_df, val_df, test_df, test_comp_df = load_processed_data()
        else:
            raise FileNotFoundError("Forcing regeneration of data")
    except (FileNotFoundError, Exception) as e:
        print(f"[WARNING] Could not load pre-processed data: {e}")
        print("[INFO] Regenerating data from raw sources...")
        master_zip = "data/raw/msk-redefining-cancer-treatment.zip"
        if not os.path.exists(master_zip):
            raise FileNotFoundError(f"Master zip not found at {master_zip}")
        train_df, val_df, test_df, test_comp_df = load_all_data(
            extract_master=True,
            master_zip_path=master_zip
        )
    
    # 2. Get Models
    print("[INFO] Loading models...")
    models = get_models()
    
    # 3. Get Preprocessing Configurations
    print("[INFO] Generating preprocessing configurations...")
    if sample_configs is None:
        all_configs = get_all_preprocessing_configs()
    else:
        all_configs = sample_configs
    
    print(f"[INFO] Total configurations to test: {len(all_configs)}")
    
    # 4. Create output directory structure
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    results_dir = output_dir / "results_by_config"
    results_dir.mkdir(exist_ok=True)
    
    # 5. Logger for TensorBoard
    logger = TBLogger(experiment_name="preprocessing_experiment")
    
    # Results tracking
    all_results = {}
    errors = []
    
    # 6. Iterate through preprocessing configurations
    config_pbar = tqdm(all_configs, desc="Preprocessing Configs")
    
    for config_idx, config in enumerate(config_pbar):
        config_pbar.set_postfix_str(str(config))
        config_name = (f"cat-{config.categorical_encoding}_"
                      f"tgt-{config.categorical_target}_"
                      f"txt-{config.text_method}_"
                      f"bal-{config.balancing_method}")
        
        print(f"\n{'='*80}")
        print(f"[CONFIG {config_idx + 1}/{len(all_configs)}] {config_name}")
        print(f"{'='*80}\n")
        
        try:
            # Apply preprocessing with this configuration
            print(f"[INFO] Preprocessing with config: {config}")
            start_preprocess = time.time()
            
            X_train, X_val, X_test, y_train, y_val, y_test = preprocess_pipeline(
                train_df, val_df, test_df, config=config
            )
            
            preprocess_time = time.time() - start_preprocess
            print(f"[INFO] Preprocessing completed in {preprocess_time:.2f}s\n")
            
            # Store results for this config
            config_results = {
                "config": str(config),
                "models": {},
                "preprocess_time": preprocess_time
            }
            
            # Train models with this configuration
            model_pbar = tqdm(models.items(), desc="Models", leave=False)
            
            for model_name, model in model_pbar:
                model_pbar.set_postfix(model=model_name)
                print(f"\n[MODEL] Training {model_name}...")
                
                try:
                    start_time = time.time()
                    
                    # Carbon Tracker per model
                    tracker = CarbonTracker(
                        experiment_name=f"model_{model_name}_config_{config_idx}"
                    )
                    tracker.start()
                    
                    # Train
                    model.fit(X_train, y_train)
                    
                    # Stop Carbon Tracker
                    emissions = tracker.stop()
                    print(f"[INFO] Carbon Emissions for {model_name}: {emissions:.6f} kg CO2")
                    
                    # Evaluate all splits
                    all_metrics = evaluate_all(model, X_train, y_train, X_val, y_val, X_test, y_test)
                    print_metrics(all_metrics, model_name=model_name)
                    
                    # Confusion Matrix (validation)
                    cm = get_confusion_matrix(model, X_val, y_val)
                    print(f"\n[INFO] Confusion Matrix for {model_name} (Validation):\n{cm}")
                    
                    # Per-class accuracy (validation)
                    pca = per_class_accuracy(model, X_val, y_val)
                    print(f"\n[INFO] Per-Class Accuracy for {model_name} (Validation):\n{pca}")
                    
                    # Log to TensorBoard
                    logger.log_metrics(
                        f"{model_name}_config_{config_idx}",
                        {"emissions": emissions},
                        split="CARBON",
                        step=0
                    )
                    
                    # Log metrics for each split
                    for split, split_metrics in all_metrics.items():
                        logger.log_metrics(
                            f"{model_name}_config_{config_idx}",
                            split_metrics,
                            split=split.upper(),
                            step=0
                        )
                    
                    # Store model metrics
                    model_metrics = all_metrics["validation"]
                    model_metrics["carbon_emissions"] = emissions
                    config_results["models"][model_name] = model_metrics
                    
                    # Save model
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    model_filename = f"{model_name}_{config_idx}_{timestamp}.joblib"
                    config_model_dir = output_dir / "models" / config_name
                    config_model_dir.mkdir(parents=True, exist_ok=True)
                    model_path = config_model_dir / model_filename
                    joblib.dump(model, model_path)
                    print(f"[INFO] Saved model to {model_path}")
                    
                    elapsed_time = time.time() - start_time
                    print(f"[INFO] Finished {model_name} in {elapsed_time:.2f} seconds.")
                    
                except Exception as e:
                    error_msg = f"Error training {model_name} with config {config_idx}: {str(e)}"
                    print(f"[ERROR] {error_msg}")
                    errors.append(error_msg)
                    config_results["models"][model_name] = {"error": str(e)}
            
            all_results[config_name] = config_results
            
            # Save results for this config
            config_results_path = results_dir / f"config_{config_idx}_{config_name}.json"
            with open(config_results_path, 'w') as f:
                json.dump(config_results, f, indent=2)
            
        except Exception as e:
            error_msg = f"Error preprocessing with config {config_idx}: {str(e)}"
            print(f"[ERROR] {error_msg}")
            errors.append(error_msg)
            all_results[f"config_{config_idx}"] = {"error": str(e)}
    
    logger.close()
    
    # 7. Save all results
    print(f"\n{'='*80}")
    print("[INFO] Training completed!")
    print(f"{'='*80}\n")
    
    results_summary_path = output_dir / "results_summary.json"
    with open(results_summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"[INFO] Results saved to {results_summary_path}")
    
    # Print errors if any
    if errors:
        print(f"\n[WARNING] {len(errors)} errors occurred during training:")
        for error in errors:
            print(f"  - {error}")
    
    # Print summary statistics
    print(f"\n[INFO] Summary Statistics:")
    print(f"  Total configs tested: {len(all_configs)}")
    print(f"  Total models: {len(models)}")
    print(f"  Total training runs: {len(all_configs) * len(models)}")
    print(f"  Successful runs: {len(all_configs) * len(models) - len(errors)}")
    print(f"  Failed runs: {len(errors)}")
    
    # Find best configuration
    print(f"\n[INFO] Finding best performing configurations...")
    best_by_metric = {}
    for config_name, config_data in all_results.items():
        if "models" in config_data:
            for model_name, metrics in config_data["models"].items():
                if "error" not in metrics:
                    for metric_name, metric_value in metrics.items():
                        if metric_name not in best_by_metric:
                            best_by_metric[metric_name] = {
                                "value": metric_value,
                                "config": config_name,
                                "model": model_name
                            }
                        else:
                            if metric_value > best_by_metric[metric_name]["value"]:
                                best_by_metric[metric_name] = {
                                    "value": metric_value,
                                    "config": config_name,
                                    "model": model_name
                                }
    
    print("\n[INFO] Best configurations by metric:")
    for metric, best in best_by_metric.items():
        print(f"  {metric}: {best['value']:.4f} (Config: {best['config']}, Model: {best['model']})")


if __name__ == "__main__":
    # Uncomment the line below to use only a sample of configs for quick testing
    # from src.feature.preprocessing import PreprocessingConfig
    # sample_configs = [
    #     PreprocessingConfig(categorical_encoding="tfidf", text_method="tfidf", categorical_target="Gene"),
    #     PreprocessingConfig(categorical_encoding="target", text_method="word2vec", categorical_target="both")
    # ]
    # main(sample_configs=sample_configs)
    
    main(use_existing_data=True)