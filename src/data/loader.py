import pandas as pd
import zipfile
from pathlib import Path
import io
from tqdm.auto import tqdm
import os

# Helper: Read CSV from ZIP
def _read_from_zip(zip_path, sep=",", names=None, skiprows=None):
    """
    Reads a single file inside a zip archive into a pandas DataFrame.
    Assumes only one file inside the zip.
    """
    with zipfile.ZipFile(zip_path, 'r') as z:
        file_name = z.namelist()[0]

        with z.open(file_name) as f:
            return pd.read_csv(
                f,
                sep=sep,
                names=names,
                skiprows=skiprows,
                engine="python"  # needed for multi-char separators like '||'
            )


def _extract_nested_zips(raw_dir, master_zip_path):
    """
    Extract the master zip file and then extract nested zip files.
    Expected structure: msk-redefining-cancer-treatment.zip contains:
    - test_text.zip
    - test_variants.zip
    - training_text.zip
    - training_variants.zip
    """
    raw_dir = Path(raw_dir)
    
    with tqdm(desc="Extracting nested zip files", leave=False) as pbar:
        # Extract master zip
        pbar.set_postfix_str("Extracting master zip")
        with zipfile.ZipFile(master_zip_path, 'r') as z:
            z.extractall(raw_dir)
        pbar.update(1)
        
        # Extract nested zips
        nested_zips = [
            raw_dir / "test_text.zip",
            raw_dir / "test_variants.zip",
            raw_dir / "training_text.zip",
            raw_dir / "training_variants.zip"
        ]
        
        for zip_file in nested_zips:
            if zip_file.exists():
                pbar.set_postfix_str(f"Extracting {zip_file.name}")
                with zipfile.ZipFile(zip_file, 'r') as z:
                    z.extractall(raw_dir)
                pbar.update(1)


# Load Training Data
def load_training_data(data_dir="data/raw/"):
    data_dir = Path(data_dir)

    variants_file = data_dir / "training_variants"
    text_file = data_dir / "training_text"

    with tqdm(total=3, desc="Loading Training Data", leave=False) as pbar:
        # Variants (normal CSV)
        pbar.set_postfix_str("variants")
        variants_df = pd.read_csv(variants_file)
        pbar.update(1)

        # Text (special separator)
        pbar.set_postfix_str("text")
        text_df = pd.read_csv(
            text_file,
            sep=r"\|\|",
            names=["ID", "TEXT"],
            skiprows=1,
            engine="python"
        )
        pbar.update(1)

        # Merge
        pbar.set_postfix_str("merging")
        df = variants_df.merge(text_df, on="ID", how="left")
        pbar.update(1)

    return df


# Load Test Data
def load_test_data(data_dir="data/raw"):
    data_dir = Path(data_dir)

    variants_file = data_dir / "test_variants"
    text_file = data_dir / "test_text"

    with tqdm(total=3, desc="Loading Test Data", leave=False) as pbar:
        # Variants
        pbar.set_postfix_str("variants")
        variants_df = pd.read_csv(variants_file)
        pbar.update(1)

        # Text
        pbar.set_postfix_str("text")
        text_df = pd.read_csv(
            text_file,
            sep=r"\|\|",
            names=["ID", "TEXT"],
            skiprows=1,
            engine="python"
        )
        pbar.update(1)

        # Merge
        pbar.set_postfix_str("merging")
        df = variants_df.merge(text_df, on="ID", how="left")
        pbar.update(1)

    return df


# Combined Loader with Split into Train/Validation/Test
def load_all_data(data_dir="data/raw", processed_dir="data/processed", extract_master=False, master_zip_path=None):
    """
    Load and prepare data with the following workflow:
    1. Optionally extract nested zip files from master zip
    2. Load training data
    3. Load test data
    4. Split training data into train/validation/test (80/10/10)
    5. Save to data/processed as CSV files
    
    Args:
        data_dir: Path to raw data directory
        processed_dir: Path to save processed data
        extract_master: Whether to extract master zip file
        master_zip_path: Path to master zip file if extraction needed
        
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    from sklearn.model_selection import train_test_split
    
    data_dir = Path(data_dir)
    processed_dir = Path(processed_dir)
    
    # Create processed directory if it doesn't exist
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract nested zips if requested
    if extract_master and master_zip_path:
        print("[INFO] Extracting master zip file...")
        _extract_nested_zips(data_dir, master_zip_path)
    
    # Load data
    print("[INFO] Loading data...")
    with tqdm(total=2, desc="Overall Data Loading Progress") as pbar:
        train_full_df = load_training_data(data_dir)
        pbar.update(1)
        test_df = load_test_data(data_dir)
        pbar.update(1)
    
    # Split training data into train/val/test (80/10/10)
    print("[INFO] Splitting training data into train/validation/test...")
    y = train_full_df['Class'].values
    
    # First split: 80% train, 20% temp
    train_df, temp_df, _, y_temp = train_test_split(
        train_full_df,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )
    
    # Second split: split 20% into 50/50 (10% val, 10% test)
    val_df, val_test_df, _, _ = train_test_split(
        temp_df,
        y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=42
    )
    
    # Save to processed directory
    print("[INFO] Saving processed data...")
    train_df.to_csv(processed_dir / "train.csv", index=False)
    val_df.to_csv(processed_dir / "validation.csv", index=False)
    val_test_df.to_csv(processed_dir / "train_test.csv", index=False)
    test_df.to_csv(processed_dir / "test.csv", index=False)
    
    print(f"[INFO] Train shape: {train_df.shape}")
    print(f"[INFO] Validation shape: {val_df.shape}")
    print(f"[INFO] Test (from training) shape: {val_test_df.shape}")
    print(f"[INFO] Test shape: {test_df.shape}")
    
    return train_df, val_df, test_df, val_test_df


def load_processed_data(processed_dir="data/processed"):
    """
    Load pre-processed data from CSV files.
    
    Args:
        processed_dir: Path to processed data directory
        
    Returns:
        tuple: (train_df, val_df, test_df, test_competition_df)
    """
    processed_dir = Path(processed_dir)
    
    train_df = pd.read_csv(processed_dir / "train.csv")
    val_df = pd.read_csv(processed_dir / "validation.csv")
    test_df = pd.read_csv(processed_dir / "train_test.csv")
    test_competition_df = pd.read_csv(processed_dir / "test.csv")
    
    return train_df, val_df, test_df, test_competition_df


"""
How to use it:

# First time: extract and process
from src.data.loader import load_all_data
train_df, val_df, test_df, test_comp_df = load_all_data(
    extract_master=True,
    master_zip_path='data/raw/msk-redefining-cancer-treatment.zip'
)

# Subsequent times: load processed data directly
from src.data.loader import load_processed_data
train_df, val_df, test_df, test_comp_df = load_processed_data()
"""