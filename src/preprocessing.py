# src/preprocessing.py
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
DROP_COLUMNS = ["X2", "X3", "X4", "X5", "X7", "X8", "X9", "X10", "X11", "X12", "X13", "X14", "X15", "X16", "X17", "X19", "X20", "X21", "X22", "X23"]


@dataclass
class SplitData:
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
    Split dataframe into train/validation/test sets.
    Default split is 70/15/15.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe.")

    df = df.copy()

    for col in DROP_COLUMNS:
        if col in df.columns:
            df = df.drop(columns=col)

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
    known_categorical = ["X2", "X3", "X4"]
    categorical_cols = [col for col in known_categorical if col in X.columns]
    numerical_cols = [col for col in X.columns if col not in categorical_cols]
    return categorical_cols, numerical_cols


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """
    Build preprocessing pipeline:
    - impute missing categorical values with most frequent
    - one-hot encode categorical variables
    - impute missing numerical values with median
    - scale numerical variables
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
    Fit preprocessor on training data only, then transform val/test.
    This avoids data leakage.
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

BASE_DIR = Path().resolve().parent

def save_processed_data(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    preprocessor: ColumnTransformer,
    output_dir: str | Path = BASE_DIR / "data" / "processed",
    pipeline_path: str | Path = BASE_DIR / "models"/ "preprocessor.pkl",
) -> None:
    """
    Save processed datasets and fitted preprocessing pipeline.
    """
    output_dir = Path(output_dir)
    pipeline_path = Path(pipeline_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline_path.parent.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(output_dir / "X_train.csv", index=False)
    X_val.to_csv(output_dir / "X_val.csv", index=False)
    X_test.to_csv(output_dir / "X_test.csv", index=False)

    y_train.to_csv(output_dir / "y_train.csv", index=False)
    y_val.to_csv(output_dir / "y_val.csv", index=False)
    y_test.to_csv(output_dir / "y_test.csv", index=False)

    joblib.dump(preprocessor, pipeline_path)


def run_preprocessing(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    random_state: int = 42,
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Full preprocessing pipeline:
    1. split data
    2. fit preprocessing on train only
    3. transform val/test
    4. optionally save outputs
    """
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
        )

    return (
        X_train_processed,
        X_val_processed,
        X_test_processed,
        split.y_train,
        split.y_val,
        split.y_test,
    )