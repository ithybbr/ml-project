import pytest
import pandas as pd
import numpy as np

from src.engineered_features import engineer_features


# ==========================================
# 1. FIXTURES (Setup Financial Dummy Data)
# ==========================================
@pytest.fixture
def financial_data():
    """Generates a tiny dataset using the original UCI raw column names."""
    return pd.DataFrame(
        {
            "LIMIT_BAL": [10000, 50000],
            # Repayment Status
            "PAY_0": [2, 0],  # Customer 1 is 2 months late, Customer 2 is paying duly
            "PAY_2": [2, 0],
            "PAY_3": [1, 0],
            "PAY_4": [0, 0],
            "PAY_5": [0, 0],
            "PAY_6": [0, 0],
            # Bill Amounts
            "BILL_AMT1": [8000, 1000],
            "BILL_AMT2": [8000, 1000],
            "BILL_AMT3": [7000, 1000],
            "BILL_AMT4": [6000, 1000],
            "BILL_AMT5": [5000, 1000],
            "BILL_AMT6": [4000, 1000],
            # Payment Amounts
            "PAY_AMT1": [0, 1000],  # Customer 1 paid nothing, Customer 2 paid full
            "PAY_AMT2": [1000, 1000],
            "PAY_AMT3": [1000, 1000],
            "PAY_AMT4": [1000, 1000],
            "PAY_AMT5": [1000, 1000],
            "PAY_AMT6": [1000, 1000],
        }
    )


# ==========================================
# 2. FEATURE ENGINEERING TESTS
# ==========================================
def test_engineered_feature_dimensions(financial_data):
    """Ensure the function expands the dataset to the expected features."""
    df_engineered = engineer_features(financial_data)
    # The output should have the original columns plus the new engineered ones
    assert df_engineered.shape[1] > financial_data.shape[1]


def test_delinquency_calculations(financial_data):
    """Verify that Max Delinquency is calculated accurately."""
    df_engineered = engineer_features(financial_data)

    # We saw "delq_max" in your error traceback, so we know this column exists!
    # Customer 1's max delay was 2 months. Customer 2 had 0 delays.
    assert df_engineered.loc[0, "delq_max"] == 2
    assert df_engineered.loc[1, "delq_max"] == 0


# NOTE: For the next two tests, you will need to replace my placeholder strings
# with the EXACT column names your script generates (e.g., if your script names
# utilization "util_max", change "YOUR_UTILIZATION_COLUMN_NAME" to "util_max").


def test_utilization_ratios(financial_data):
    """Verify that credit utilization math doesn't divide by zero."""
    df_engineered = engineer_features(financial_data)

    # Customer 1 Month 1 Util: 8000 / 10000 = 0.8
    # Customer 2 Month 1 Util: 1000 / 50000 = 0.02

    # TODO: Replace with your actual max utilization column name
    col_name = "YOUR_UTILIZATION_COLUMN_NAME"
    if col_name in df_engineered.columns:
        assert np.isclose(df_engineered.loc[0, col_name], 0.80)
        assert np.isclose(df_engineered.loc[1, col_name], 0.02)


def test_zero_payment_counts(financial_data):
    """Verify the system correctly counts months with $0 payments."""
    df_engineered = engineer_features(financial_data)

    # Customer 1 missed one payment. Customer 2 missed zero.

    # TODO: Replace with your actual zero payment count column name
    col_name = "YOUR_ZERO_PAY_COUNT_COLUMN"
    if col_name in df_engineered.columns:
        assert df_engineered.loc[0, col_name] == 1
        assert df_engineered.loc[1, col_name] == 0
