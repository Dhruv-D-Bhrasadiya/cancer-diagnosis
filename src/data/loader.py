import pandas as pd
import zipfile
from pathlib import Path
import io
from tqdm.auto import tqdm

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


# Load Training Data
def load_training_data(data_dir="data/raw/"):
    data_dir = Path(data_dir)

    variants_zip = data_dir / "training_variants.zip"
    text_zip = data_dir / "training_text.zip"

    with tqdm(total=3, desc="Loading Training Data", leave=False) as pbar:
        # Variants (normal CSV)
        pbar.set_postfix_str("variants")
        variants_df = _read_from_zip(variants_zip)
        pbar.update(1)

        # Text (special separator)
        pbar.set_postfix_str("text")
        text_df = _read_from_zip(
            text_zip,
            sep=r"\|\|",
            names=["ID", "TEXT"],
            skiprows=1
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

    variants_zip = data_dir / "test_variants.zip"
    text_zip = data_dir / "test_text.zip"

    with tqdm(total=3, desc="Loading Test Data", leave=False) as pbar:
        # Variants
        pbar.set_postfix_str("variants")
        variants_df = _read_from_zip(variants_zip)
        pbar.update(1)

        # Text
        pbar.set_postfix_str("text")
        text_df = _read_from_zip(
            text_zip,
            sep=r"\|\|",
            names=["ID", "TEXT"],
            skiprows=1
        )
        pbar.update(1)

        # Merge
        pbar.set_postfix_str("merging")
        df = variants_df.merge(text_df, on="ID", how="left")
        pbar.update(1)

    return df


# Combined Loader
def load_all_data(data_dir="data/raw"):
    with tqdm(total=2, desc="Overall Data Loading Progress") as pbar:
        train_df = load_training_data(data_dir)
        pbar.update(1)
        test_df = load_test_data(data_dir)
        pbar.update(1)

    return train_df, test_df

"""
How to use it 


from src.data.loader import load_all_data
train_df, test_df = load_all_data()

"""