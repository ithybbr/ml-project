import pytest
import pandas as pd
from src.preprocessing import split_data, detect_column_types


# 1. Create a tiny "dummy" dataset to run fast tests on
@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "X1": [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000],
            "X2": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            "Y": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )


# 2. Test the splitting math
def test_split_data_shapes(sample_data):
    """Ensure the train/val/test split returns the correct mathematical proportions."""
    split = split_data(
        df=sample_data,
        target_column="Y",
        test_size=0.20,
        val_size=0.20,
        random_state=42,
    )

    # Out of 10 rows, 60% train, 20% val, 20% test
    assert len(split.X_train) == 6
    assert len(split.X_val) == 2
    assert len(split.X_test) == 2

    # Ensure the target column was actually dropped from the features
    assert "Y" not in split.X_train.columns
    assert len(split.y_train) == 6


# 3. Test the categorical detection logic
def test_detect_column_types(sample_data):
    """Ensure demographic columns are properly flagged as categorical."""
    X = sample_data.drop(columns=["Y"])
    cat_cols, num_cols = detect_column_types(X)

    assert "X2" in cat_cols
    assert "X1" in num_cols
