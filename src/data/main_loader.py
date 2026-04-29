import zipfile
import os
from pathlib import Path
from tqdm.auto import tqdm



def unzipMain(FILE_PATH: str = "data/raw/msk-redefining-cancer-treatment.zip", dir: str = "data/raw"):
    """
    Extract the master zip file and then extract nested zip files.
    Make sure there exist the main zip file, for more hints check `data/Instructions.md`
    Structure: msk-redefining-cancer-treatment.zip contains:
    - test_text.zip
    - test_variants.zip
    - training_text.zip
    - training_variants.zip
    """

    print("\n[INFO] Extracting zip files\n")

    dir = Path(dir)
    FILE_PATH = Path(FILE_PATH)

    # Extract master zip
    if(FILE_PATH.exists()):
        with zipfile.ZipFile(FILE_PATH, 'r') as z:
            z.extractall(dir) 
    else:
        print("[INFO] File not found\n")
        print("[INFO] HINT: Check `data/Instructions.md`\n")

        return

    # Extract nested zips
    nested_zips = [
        dir / "test_text.zip",
        dir / "test_variants.zip",
        dir / "training_text.zip",
        dir / "training_variants.zip"
    ]

    # Temparary Files that comes with the zip file ad
    temp_zips = [
        dir / "stage_2_private_solution.csv.7z",
        dir / "stage1_solution_filtered.csv.7z",
        dir / "stage2_sample_submission.csv.7z",
        dir / "stage2_test_text.csv.7z",
        dir / "stage2_test_variants.csv.7z"
    ]

    print("[INFO] Extracting nested zips\n")

    # Unzip the files necessary 
    for zip_file in nested_zips:
        if zip_file.exists():
            with zipfile.ZipFile(zip_file, 'r') as z:
                z.extractall(dir)

    print("[INFO] Deleting nested zips\n")

    # delete unnecessary files and the zip files which are extracted
    for file in nested_zips + temp_zips:
         os.remove(file)