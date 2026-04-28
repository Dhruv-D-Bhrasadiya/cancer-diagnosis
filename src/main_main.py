import yaml
import os
import pandas as pd

from data.main_loader import unzipMain
from feature.create_csv import create_csv
from feature.clean import OHE, TargetEncoding, BinaryEncoding, LabelEncoding, TextVectorization
from models.registry import get_models

if __name__ == "__main__":
    # Step 1: Unzip the data
    unzipMain()

    # Step 2: Create CSV files from the unzipped data
    create_csv()

    # Step 3: Create dataframes from the CSV files
    train_df = pd.read_csv("data/train.csv")
    val_df = pd.read_csv("data/val.csv")

    # Step 4: Feature Engineering
    config_path = os.path.join(os.path.dirname(__file__), "config/config.yaml")

    if not os.path.exists(config_path):
        # Try relative to src directory
        config_path = os.path.join(os.path.dirname(__file__), "../../src/config/config.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    preprocessing_config = config_data.get('preprocessing', {})

    categorical_encodings = preprocessing_config.get('categorical_encodings', ['tfidf'])
    categorical_targets = preprocessing_config.get('categorical_targets', ['Gene'])
    text_methods = preprocessing_config.get('text_methods', ['tfidf'])
    balancing_methods = preprocessing_config.get('balancing_methods', ['none'])

    # Apply encodings based on config
    if 'ohe' in categorical_encodings:
        ohe = OHE(train_df)
        train_df = pd.concat([train_df, ohe.ohe_gene(train_df), ohe.ohe_variation(train_df), ohe.ohe_class(train_df)], axis=1)
    if 'target' in categorical_encodings:
        te = TargetEncoding(train_df)
        for target in categorical_targets:
            train_df[f"{target}_te"] = te.target_encode(target, train_df['Class'])
    if 'binary' in categorical_encodings:
        be = BinaryEncoding(train_df)
        train_df = pd.concat([train_df, be.binary_encode("Gene"), be.binary_encode("Variation"), be.binary_encode("Class")], axis=1)