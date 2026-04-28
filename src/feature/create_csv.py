import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def create_csv(BASE_DIR: str = "./data/raw", OUT_DIR: str = "./data/processed"):

    # Create path and make dir
    BASE_DIR = Path(BASE_DIR)
    OUT_DIR = Path(OUT_DIR)
    OUT_DIR.mkdir(parents = True, exist_ok = True)

    # Assign file paths
    train_var_path = BASE_DIR / "training_variants"
    test_var_path = BASE_DIR / "test_variants"
    train_text_path = BASE_DIR / "training_text"
    test_text_path = BASE_DIR / "test_text"

    required_file = [
        train_var_path, test_var_path,
        train_text_path, test_text_path
    ]


    for f in required_file:
        if not f.exists():
            print(f"\n[ERROR] Missing files. Please read data/instructions.md\n")
            return

    print("[INFO] Loading variants file")

    # Load variants file
    train_var = pd.read_csv(train_var_path)
    test_var = pd.read_csv(test_var_path)

    print("[INFO] Loading text file")

    # Load text file
    train_text = pd.read_csv(
            train_text_path,
            sep=r"\|\|",
            names=["ID", "TEXT"],
            skiprows=1,
            engine="python"
        )
    test_text = pd.read_csv(
            test_text_path,
            sep=r"\|\|",
            names=["ID", "TEXT"],
            skiprows=1,
            engine="python"
        )
    
    print(test_text.head)
    print(train_text.head)

    print("[INFO] Merging data")

    # Merge variants and text file
    train_df = pd.merge(train_var, train_text, on = "ID", how = "inner")
    test_df = pd.merge(test_var, test_text, on = "ID", how = "inner")

    print("[INFO] Spliting data")

    # Create Validation split
    train_df, val_df = train_test_split(
        train_df,
        test_size = 0.1,
        stratify = train_df["Class"],
        random_state = 42
    )

    print("[INFO] Saving to CSV")

    print(train_var["ID"].dtype, train_text["ID"].dtype)

    # Save to csv files 
    train_df.to_csv(OUT_DIR / "train_data.csv", index=False)
    test_df.to_csv(OUT_DIR / "test_data.csv", index=False)
    val_df.to_csv(OUT_DIR / "val_data.csv", index=False)

    # Print Info
    print("[INFO] All files Save to data/processed")


create_csv()