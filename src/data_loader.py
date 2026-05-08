"""
Data Loading Utilities.

Provides functions for loading the raw Excel data and handling the specific
double-header formatting present in the UCI dataset to ensure clean ingestion.
"""

from pathlib import Path
from typing import Union
import pandas as pd


def load_data(path: Union[str, Path]) -> tuple[pd.DataFrame, pd.Series]:
    """
    Loads raw Excel data, removes the ID column, and extracts the feature map header.

    Args:
        path (str | Path): The file path to the raw .xls or .xlsx dataset.

    Returns:
        tuple[pd.DataFrame, pd.Series]: A tuple containing:
            - pd.DataFrame: The cleaned dataset containing only numerical data.
            - pd.Series: The extracted feature map (originally located in row 0).
    """
    df = pd.read_excel(path)
    df.rename(columns={df.columns[0]: "ID"}, inplace=True)
    df.drop(columns=["ID"], inplace=True)

    features = df.iloc[0]
    df.drop(index=0, inplace=True)

    return df, features
