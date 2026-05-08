"""
Dataset Downloader.

Automates downloading the raw 'Default of Credit Card Clients' dataset
directly from the UCI Machine Learning Repository, extracting the zip archive,
and standardizing the filename for the rest of the pipeline.
"""

import urllib.request
import zipfile
import os
from pathlib import Path

# --- Configuration ---
URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"

# Assuming this script is placed in the 'src' directory, BASE_DIR becomes the project root
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
ZIP_PATH = RAW_DIR / "temp_dataset.zip"
FINAL_FILE = RAW_DIR / "data.xls"


def main() -> None:
    """
    Downloads the dataset zip file, extracts its contents, renames the target Excel file
    to 'data.xls', and cleans up temporary archive files.
    """
    # 1. Ensure the target directory exists
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Download the zip file
    print(f"📥 Downloading dataset from {URL}...")
    try:
        urllib.request.urlretrieve(URL, ZIP_PATH)
        print("✅ Download complete.")
    except Exception as e:
        print(f"❌ Failed to download file: {e}")
        return

    # 3. Extract the contents
    print("📦 Extracting files...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        extracted_files = zip_ref.namelist()
        zip_ref.extractall(RAW_DIR)

    # 4. Find the extracted .xls file, rename it, and move it
    for file_name in extracted_files:
        if file_name.endswith(".xls") or file_name.endswith(".xlsx"):
            original_extracted_path = RAW_DIR / file_name

            # If an older data.xls already exists, remove it first
            if FINAL_FILE.exists():
                FINAL_FILE.unlink()

            original_extracted_path.rename(FINAL_FILE)
            print(f"🔄 Renamed '{file_name}' to 'data.xls'")
            break

    # 5. Clean up the leftover zip file
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
        print("🧹 Cleaned up temporary zip file.")

    print(f"\n🎉 Success! The raw dataset is ready at: {FINAL_FILE}")


if __name__ == "__main__":
    main()
