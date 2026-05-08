"""
Model Calibration Analysis Module.

Evaluates the reliability of predicted probabilities across multiple machine
learning models. Generates calibration curves (reliability diagrams) and calculates
the Brier Score to ensure that predicted risk percentages align with actual default rates.
"""

from __future__ import annotations

from pathlib import Path
import joblib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import seaborn as sns

sns.set_theme(style="whitegrid")


def analyze_calibration(
    models_dict: dict[str, Path], data_path: Path, output_dir: Path
) -> pd.DataFrame:
    """
    Evaluates probability calibration for a suite of models and generates visualization plots.

    Args:
        models_dict (dict[str, Path]): A dictionary mapping model names to their .pkl file paths.
        data_path (Path): Path to the processed test data .pkl file.
        output_dir (Path): Directory where the output plots and CSVs will be saved.

    Returns:
        pd.DataFrame: A dataframe containing the Brier Score for each evaluated model.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load test data
    data = joblib.load(data_path)
    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = data[:7]
    y_test = y_test.astype(int)

    plt.figure(figsize=(10, 8))

    # Plot the perfect calibration line
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")

    brier_scores = []

    for name, path in models_dict.items():
        if not path.exists():
            print(f"Skipping {name}: Model file not found.")
            continue

        model = joblib.load(path)

        # Ensure model has predict_proba
        if not hasattr(model, "predict_proba"):
            print(f"Skipping {name}: Model does not support predict_proba.")
            continue

        # Get probabilities for the positive class (Default = 1)
        prob_pos = model.predict_proba(X_test)[:, 1]

        # Calculate Brier Score
        brier = brier_score_loss(y_test, prob_pos)
        brier_scores.append({"Model": name, "Brier_Score": brier})

        # Calculate Calibration Curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, prob_pos, n_bins=10, strategy="uniform"
        )

        # Add to plot
        plt.plot(
            mean_predicted_value,
            fraction_of_positives,
            "s-",
            label=f"{name} (Brier: {brier:.3f})",
        )

    # Format Plot
    plt.xlabel("Mean Predicted Probability (Risk Score)")
    plt.ylabel("Fraction of Actual Defaults")
    plt.title("Calibration Curves (Reliability Diagrams)")
    plt.legend(loc="lower right")
    plt.tight_layout()

    # Save Plot
    plot_path = output_dir / "calibration_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Calibration plot saved to {plot_path}")

    # Save Scores
    scores_df = pd.DataFrame(brier_scores).sort_values("Brier_Score")
    scores_path = output_dir / "brier_scores.csv"
    scores_df.to_csv(scores_path, index=False)
    print(f"Brier scores saved to {scores_path}")

    return scores_df


def main() -> None:
    """Executes the calibration analysis sequence."""
    models_dir = Path("models")
    output_dir = Path("results/calibration")

    # Select the 44-feature models for comprehensive analysis
    target_models = {
        "LightGBM": models_dir / "lightgbm_44features.pkl",
        "XGBoost": models_dir / "xgboost_44features.pkl",
        "Random Forest": models_dir / "random_forest_44features.pkl",
        "Logistic Regression": models_dir / "logreg_44features.pkl",
    }

    data_path = Path("data/processed/44features.pkl")

    if not data_path.exists():
        print(f"Error: Data file {data_path} not found.")
        return

    print(f"\n{'='*60}")
    print("CALIBRATION ANALYSIS")
    print(f"{'='*60}")

    analyze_calibration(target_models, data_path, output_dir)


if __name__ == "__main__":
    main()
