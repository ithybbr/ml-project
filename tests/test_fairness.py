import pytest
import numpy as np
import pandas as pd
from src.fairness_analysis import subgroup_metrics


def test_subgroup_metrics_calculations():
    """Verify that the fairness math (Recall, FPR, FNR) is calculated accurately per group."""

    # 1. Setup Deterministic Data
    # Total 8 people. 4 defaults (1), 4 safe (0).
    y_true = pd.Series([1, 1, 0, 0, 1, 1, 0, 0])

    # The model perfectly predicts Group 1, but is terrible at predicting Group 2
    y_pred = np.array([1, 1, 0, 0, 0, 0, 0, 0])

    # Group 1 = Male (1), Group 2 = Female (2)
    demo_df = pd.DataFrame({"SEX": [1, 1, 1, 1, 2, 2, 2, 2]})

    group_names = {1: "Male", 2: "Female"}

    # 2. Run the calculation
    results_df = subgroup_metrics(
        y_true, y_pred, demo_df, group_col="SEX", group_names=group_names
    )

    # 3. Extract the isolated stats for each group
    male_stats = results_df[results_df["Group"] == "Male"].iloc[0]
    female_stats = results_df[results_df["Group"] == "Female"].iloc[0]

    # 4. Verify the math
    assert male_stats["Count"] == 4

    # Group 1 (Male) had 2 actual defaults, and the model found both of them.
    assert male_stats["Recall"] == 1.0
    assert male_stats["Accuracy"] == 1.0

    # Group 2 (Female) had 2 actual defaults, but the model found NONE of them.
    assert female_stats["Recall"] == 0.0

    # False Negative Rate (FNR) for females should be 100% since it missed all defaults
    assert female_stats["FNR"] == 1.0

    # The approval rate for females should be 100% (since y_pred is all 0s for them)
    assert female_stats["Approval_Rate"] == 1.0
