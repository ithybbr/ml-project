import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from pathlib import Path
from src.data_loader import load_data
import src.download_data as download_script


# ==========================================
# 1. TEST DATA LOADER (Using temporary files)
# ==========================================
def test_load_data(tmp_path):
    """Ensure the loader drops the ID column and correctly extracts the double header."""
    # 1. Create a fake Excel file in a temporary testing folder
    fake_df = pd.DataFrame(
        {
            "Unnamed: 0": ["ID", 1, 2],
            "X1": ["LIMIT_BAL", 50000, 10000],
            "X2": ["SEX", 1, 2],
        }
    )

    # tmp_path is a magic Pytest fixture that creates a temporary directory
    # that automatically deletes itself after the test runs!
    fake_excel_path = tmp_path / "fake_data.xlsx"
    fake_df.to_excel(fake_excel_path, index=False)

    # 2. Run your loader
    df, features = load_data(fake_excel_path)

    # 3. Verify the formatting logic worked
    assert "ID" not in df.columns, "The ID column was not dropped!"
    assert len(df) == 2, "The header row was not dropped from the dataframe!"
    assert (
        features["X1"] == "LIMIT_BAL"
    ), "The feature names were not extracted correctly!"
    assert df.iloc[0]["X1"] == 50000, "The data shifted incorrectly!"


# ==========================================
# 2. TEST DOWNLOADER (Using Network Mocks)
# ==========================================
@patch("src.download_data.urllib.request.urlretrieve")
@patch("src.download_data.zipfile.ZipFile")
def test_download_data_script(mock_zip, mock_urlretrieve, tmp_path):
    """Ensure the script handles the download and extraction workflow without hitting the internet."""
    # 1. Temporarily hijack the script's RAW_DIR so it saves to our temp folder instead of your real data folder
    download_script.RAW_DIR = tmp_path
    download_script.ZIP_PATH = tmp_path / "temp_dataset.zip"
    download_script.FINAL_FILE = tmp_path / "data.xls"

    # 2. Setup our fake zip file to pretend it contains a target .xls file
    mock_zip_instance = MagicMock()
    mock_zip_instance.namelist.return_value = ["some_readme.txt", "extracted_data.xls"]
    mock_zip.return_value.__enter__.return_value = mock_zip_instance

    # 3. Create the "extracted" file physically in the temp folder so the script can rename it
    (tmp_path / "extracted_data.xls").touch()

    # 4. Run the main function
    download_script.main()

    # 5. Assertions: Did it try to download? Did it rename the file properly?
    mock_urlretrieve.assert_called_once_with(
        download_script.URL, download_script.ZIP_PATH
    )
    assert (
        download_script.FINAL_FILE.exists()
    ), "The script failed to rename the extracted .xls file to data.xls!"
