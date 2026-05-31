import yaml
import os
import pandas as pd
from pathlib import Path

from data.main_loader import unzipMain
from feature.create_csv import create_csv
from feature.clean import (OHE, ResponseEncoding, TargetEncoding, BinaryEncoding,
                            LabelEncoding, FrequencyEncoding, HashingEncoding,
                            TF_IDF, FisrtLastText, TruncatedText, Word2VecText, Balancing)
from models.registry import get_models

if __name__ == "__main__":
    # Resolve project root regardless of where the script is run from
    project_root = Path(__file__).parent.parent

    # # Step 1: Unzip the data
    # unzipMain()

    # # Step 2: Create CSV files from the unzipped data
    # create_csv()

    # Step 3: Load processed CSVs
    raw_train_df = pd.read_csv(project_root / "data/processed/train_data.csv")
    raw_val_df   = pd.read_csv(project_root / "data/processed/val_data.csv")

    # ── Feature extraction helpers ───────────────────────────────────────────

    def get_categorical_features(df, encoding, target_col):
        if encoding == "onehot":
            ohe = OHE(df)
            if target_col == "Gene":
                return ohe.ohe_gene(df)
            elif target_col == "Variation":
                return ohe.ohe_variation(df)
            elif target_col == "Class":
                return ohe.ohe_class(df)

        elif encoding == "response":
            re = ResponseEncoding(df)
            return re.response_encode(target_col, df["Class"]).to_frame(name=f"{target_col}_resp")

        elif encoding == "target":
            te = TargetEncoding(df)
            return te.target_encode(target_col, df["Class"]).to_frame(name=f"{target_col}_target")

        elif encoding == "frequency":
            fe = FrequencyEncoding(df)
            return fe.frequency_encode(target_col).to_frame(name=f"{target_col}_freq")

        elif encoding == "binary":
            be = BinaryEncoding(df)
            return be.binary_encode(target_col)

        elif encoding == "hashing":
            he = HashingEncoding(df)
            return pd.DataFrame(he.hashing_encode(target_col).toarray())

        return None

    def get_text_features(df, method):
        if method == "tfidf":
            tf = TF_IDF(df)
            return pd.DataFrame(tf.tfidf_vectorize("TEXT").toarray())

        elif method == "first_last":
            fl = FisrtLastText(df)
            return fl.extract_first_last("TEXT")

        elif method == "truncated":
            tt = TruncatedText(df)
            return tt.truncate_text("TEXT").to_frame()

        elif method == "word2vec":
            w2v = Word2VecText(df)
            return pd.DataFrame(w2v.word2vec_vectorize("TEXT"))

        return None

    # ── Load config ──────────────────────────────────────────────────────────

    config_path = project_root / "src/config/config.yaml"
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        raise FileNotFoundError(f"Configuration file not found at '{config_path}'")

    categorical_encodings = config["preprocessing"]["categorical_encodings"]
    categorical_targets   = config["preprocessing"]["categorical_targets"]
    text_methods          = config["preprocessing"]["text_methods"]
    balancing_methods     = config["preprocessing"]["balancing_methods"]

    # Ensure outputs directory exists
    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # ── Main loop ────────────────────────────────────────────────────────────

    for enc in categorical_encodings:
        for col in categorical_targets:
            for text_method in text_methods:

                print(f"[INFO] Running config: {enc} | {col} | {text_method}")

                # --- TRAIN FEATURES ---
                gene_feat_train = get_categorical_features(raw_train_df, enc, col)
                text_feat_train = get_text_features(raw_train_df, text_method)

                X_train = pd.concat([gene_feat_train, text_feat_train], axis=1)
                # Convert all column names to str — required by SMOTE
                X_train.columns = X_train.columns.astype(str)
                y_train = raw_train_df["Class"]

                # --- VAL FEATURES ---
                gene_feat_val = get_categorical_features(raw_val_df, enc, col)
                text_feat_val = get_text_features(raw_val_df, text_method)

                X_val = pd.concat([gene_feat_val, text_feat_val], axis=1)
                X_val.columns = X_val.columns.astype(str)
                y_val = raw_val_df["Class"]

                # --- BALANCING ---
                for bal_method in balancing_methods:
                    X_res, y_res = X_train, y_train

                    if bal_method != "none":
                        try:
                            balancer = Balancing(raw_train_df)

                            if bal_method == "smote_oversample":
                                X_res, y_res = balancer.smote_over_sample(X_train, y_train)

                            elif bal_method == "smote_undersample":
                                X_res, y_res = balancer.smote_under_sample(X_train, y_train)

                            elif bal_method == "smote_combined":
                                X_res, y_res = balancer.smote_combined_sample(X_train, y_train)

                        except (ValueError, TypeError) as e:
                            print(f"[WARNING] Skipping {bal_method} for {enc}|{col}|{text_method} — {e}")
                            X_res, y_res = X_train, y_train

                    # --- SAVE ---
                    name = f"{enc}_{col}_{text_method}_{bal_method}"

                    pd.DataFrame(X_res).to_csv(outputs_dir / f"train_{name}.csv", index=False)
                    pd.DataFrame(X_val).to_csv(outputs_dir / f"val_{name}.csv",   index=False)

                    print(f"[INFO] Saved: {name}")
