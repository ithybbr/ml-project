"""
Data Preprocessing Pipeline Module.

This module provides utilities to split data safely, identify feature types,
impute missing values, encode categorical strings, and apply standard scaling.
It ensures strict adherence to avoiding data leakage by separating fit and transform phases.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "Y"
BASE_DIR = Path().resolve().parent


@dataclass
class SplitData:
    """A data structure for holding train, validation, and test splits."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def split_data(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> SplitData:
    """
    Splits a dataframe into training, validation, and test sets.

    Applies a stratified split based on the target column to ensure
    class distribution is maintained across all three datasets.

    Args:
        df (pd.DataFrame): The complete raw dataset.
        target_column (str, optional): The name of the label column. Defaults to "Y".
        test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.15.
        val_size (float, optional): Proportion of the dataset to include in the validation split. Defaults to 0.15.
        random_state (int, optional): Seed for reproducible shuffles. Defaults to 42.

    Raises:
        ValueError: If the specified target_column is not found in the dataframe.

    Returns:
        SplitData: A dataclass containing X_train, X_val, X_test, y_train, y_val, and y_test.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe.")

    X = df.drop(columns=target_column)
    y = df[target_column]

    # first split: train vs temp
    temp_size = test_size + val_size
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=temp_size,
        random_state=random_state,
        stratify=y,
    )

    # second split: val vs test
    relative_test_size = test_size / temp_size
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_test_size,
        random_state=random_state,
        stratify=y_temp,
    )

    return SplitData(X_train, X_val, X_test, y_train, y_val, y_test)


def detect_column_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Identifies categorical and numerical columns within the feature matrix based on predefined lists.

    Args:
        X (pd.DataFrame): The feature matrix to inspect.

    Returns:
        tuple[list[str], list[str]]: Two lists, the first containing the names of categorical columns,
            and the second containing the names of numerical columns.
    """
    known_categorical = ["X2", "X3", "X4", "X30"]
    categorical_cols = [col for col in known_categorical if col in X.columns]
    numerical_cols = [col for col in X.columns if col not in categorical_cols]
    return categorical_cols, numerical_cols


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """
    Constructs a scikit-learn ColumnTransformer pipeline targeting numeric and categorical data.

    Categorical processing involves most-frequent imputation and One-Hot Encoding.
    Numerical processing involves median imputation and Standard Scaling.

    Args:
        X_train (pd.DataFrame): The training feature matrix used to identify columns.

    Returns:
        ColumnTransformer: The un-fitted preprocessing pipeline object.
    """
    categorical_cols, numerical_cols = detect_column_types(X_train)

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )

    return preprocessor


def preprocess_and_transform(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, ColumnTransformer]:
    """
    Executes the preprocessing logic safely to prevent data leakage.

    The preprocessor is exclusively fitted to the `X_train` data, then applied
    as a transformation to the validation and test sets. Feature names are explicitly
    preserved and returned as DataFrames instead of raw numpy arrays.

    Args:
        X_train (pd.DataFrame): The training feature matrix (used for fitting).
        X_val (pd.DataFrame): The validation feature matrix.
        X_test (pd.DataFrame): The testing feature matrix.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, ColumnTransformer]: The three processed
            dataframes accompanied by the fitted ColumnTransformer object.
    """
    preprocessor = build_preprocessor(X_train)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = get_feature_names(preprocessor)

    X_train_processed = pd.DataFrame(
        X_train_processed, columns=feature_names, index=X_train.index
    )
    X_val_processed = pd.DataFrame(
        X_val_processed, columns=feature_names, index=X_val.index
    )
    X_test_processed = pd.DataFrame(
        X_test_processed, columns=feature_names, index=X_test.index
    )

    return X_train_processed, X_val_processed, X_test_processed, preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """
    Extracts the updated column names from a fitted ColumnTransformer (crucial for One-Hot Encoded variables).

    Args:
        preprocessor (ColumnTransformer): The fitted preprocessor.

    Returns:
        list[str]: A list of the output column names corresponding to the transformed array.
    """
    feature_names: list[str] = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == "remainder":
            continue

        if not columns:  # skip if no columns assigned
            continue

        if hasattr(transformer, "named_steps") and "encoder" in transformer.named_steps:
            encoder = transformer.named_steps["encoder"]
            encoded_names = encoder.get_feature_names_out(columns)
            feature_names.extend(encoded_names.tolist())
        else:
            feature_names.extend(list(columns))

    return feature_names


def save_processed_data(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    preprocessor: ColumnTransformer,
    output_dir: str | Path,
    pipeline_path: str | Path,
) -> None:
    """
    Bundles the processed arrays, targets, and the fitted pipeline into a single tuple
    and saves it to the disk as a .pkl file for downstream training consumption.

    Args:
        X_train (pd.DataFrame): Transformed training features.
        X_val (pd.DataFrame): Transformed validation features.
        X_test (pd.DataFrame): Transformed testing features.
        y_train (pd.Series): Training labels.
        y_val (pd.Series): Validation labels.
        y_test (pd.Series): Testing labels.
        preprocessor (ColumnTransformer): The fitted preprocessing pipeline.
        output_dir (str | Path): Base directory for saving.
        pipeline_path (str | Path): Explicit path for the resulting .pkl file.
    """
    output_dir = Path(output_dir)
    pipeline_path = Path(pipeline_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        (X_train, X_val, X_test, y_train, y_val, y_test, preprocessor), pipeline_path
    )


def run_preprocessing(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    drop_columns: Optional[list[str]] = None,
    random_state: int = 42,
    save: bool = True,
    path: str | Path = BASE_DIR / "data" / "processed",
    name: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Executes the comprehensive data preprocessing lifecycle sequentially:
    filtering target columns, splitting data, fitting the pipeline, transforming arrays,
    and saving the serialized artifacts to the disk.

    Args:
        df (pd.DataFrame): The raw combined dataframe.
        target_column (str, optional): Label column. Defaults to TARGET_COLUMN.
        drop_columns (Optional[list[str]], optional): Columns to exclude prior to splitting. Defaults to None.
        random_state (int, optional): Seed for split shuffles. Defaults to 42.
        save (bool, optional): Whether to write outputs to disk. Defaults to True.
        path (str | Path, optional): Output save directory. Defaults to BASE_DIR / "data" / "processed".
        name (Optional[str], optional): Filename prefix for the saved artifact. Defaults to None.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]: The
            six core DataFrames and Series arrays ready for machine learning model ingestion.
    """
    if drop_columns is None:
        drop_columns = ["X2", "X3", "X4"]

    df = df.copy()

    # Drop specified columns before splitting
    for col in drop_columns:
        if col in df.columns:
            df = df.drop(columns=col)

    split = split_data(
        df=df,
        target_column=target_column,
        test_size=0.15,
        val_size=0.15,
        random_state=random_state,
    )

    X_train_processed, X_val_processed, X_test_processed, preprocessor = (
        preprocess_and_transform(split.X_train, split.X_val, split.X_test)
    )

    if save:
        save_processed_data(
            X_train_processed,
            X_val_processed,
            X_test_processed,
            split.y_train,
            split.y_val,
            split.y_test,
            preprocessor,
            output_dir=path,
            pipeline_path=(
                Path(path) / f"{name}.pkl" if name else Path(path) / "preprocessor.pkl"
            ),
        )

    return (
        X_train_processed,
        X_val_processed,
        X_test_processed,
        split.y_train,
        split.y_val,
        split.y_test,
    )
