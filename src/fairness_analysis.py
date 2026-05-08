# src/fairness_analysis.py
"""
Fairness analysis for credit scoring models.
Evaluates model performance across demographic subgroups (SEX, EDUCATION, MARRIAGE).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 1. LOAD DEMOGRAPHICS (from demographic.pkl - raw values)
# ============================================================


def load_demographics(split: str = "test") -> pd.DataFrame:
    """
    Load raw demographic data for a specific split from demographic.pkl.

    demographic.pkl structure:
    Item 0: X_train_demo (21000, 3) - columns X2, X3, X4
    Item 1: X_val_demo   (4500, 3)
    Item 2: X_test_demo  (4500, 3)
    Item 3: y_train
    Item 4: y_val
    Item 5: y_test
    """
    demo_path = Path("data/processed/demographic.pkl")
    if not demo_path.exists():
        raise FileNotFoundError(f"{demo_path} not found.")

    demo = joblib.load(demo_path)

    if split == "train":
        return demo[0]
    elif split == "val":
        return demo[1]
    else:
        return demo[2]


# ============================================================
# 2. COMPUTE SUBGROUP METRICS
# ============================================================


def subgroup_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    demo_df: pd.DataFrame,
    group_col: str,
    group_names: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Compute fairness metrics for a demographic subgroup.
    """
    # Cast to int for proper comparison
    demo_df = demo_df.copy()
    demo_df[group_col] = demo_df[group_col].astype(int)

    results = []

    for group in sorted(demo_df[group_col].unique()):
        mask = demo_df[group_col] == group
        n = mask.sum()

        if n == 0:
            continue

        y_true_g = y_true[mask]
        y_pred_g = y_pred[mask]

        # Confusion matrix
        cm = confusion_matrix(y_true_g, y_pred_g, labels=[0, 1])
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0

        group_label = group_names.get(group, str(group)) if group_names else str(group)

        results.append(
            {
                "Group": group_label,
                "Count": n,
                "Default_Rate": y_true_g.mean(),
                "Approval_Rate": (y_pred_g == 0).mean(),
                "Accuracy": accuracy_score(y_true_g, y_pred_g),
                "Precision": precision_score(y_true_g, y_pred_g, zero_division=0),
                "Recall": recall_score(y_true_g, y_pred_g, zero_division=0),
                "F1": f1_score(y_true_g, y_pred_g, zero_division=0),
                "FPR": fp / (fp + tn) if (fp + tn) > 0 else 0,
                "FNR": fn / (fn + tp) if (fn + tp) > 0 else 0,
            }
        )

    return pd.DataFrame(results)


# ============================================================
# 3. RUN ANALYSIS FOR ONE MODEL
# ============================================================


def analyze_model(
    model_path: str | Path,
    features_path: str | Path,
    output_dir: str | Path = "results/fairness",
) -> dict[str, pd.DataFrame]:
    """
    Run complete fairness analysis for a single model.
    """
    model_path = Path(model_path)
    features_path = Path(features_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model and test data
    model = joblib.load(model_path)
    data = joblib.load(features_path)
    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = data[:7]

    y_train = y_train.astype(int)
    y_val = y_val.astype(int)
    y_test = y_test.astype(int)

    # Predictions on test set
    y_pred = model.predict(X_test).astype(int)

    # Load demographics for test set (item 2)
    demo_test = load_demographics("test")

    # Verify alignment
    assert len(demo_test) == len(
        X_test
    ), f"Demo ({len(demo_test)}) != X_test ({len(X_test)})"
    assert (demo_test.index == X_test.index).all(), "Index mismatch!"

    # Human-readable names
    sex_names = {1: "Male", 2: "Female"}
    education_names = {
        0: "Other",
        1: "Graduate School",
        2: "University",
        3: "High School",
        4: "Other",
        5: "Unknown",
        6: "Unknown",
    }
    marriage_names = {0: "Other", 1: "Married", 2: "Single", 3: "Other"}

    model_name = model_path.stem

    print(f"\n{'='*60}")
    print(f"FAIRNESS ANALYSIS: {model_name}")
    print(f"Test set: {len(y_test)} rows")
    print(f"Overall default rate: {y_test.mean():.3f}")
    print(f"{'='*60}")

    # Run analysis
    sex_df = subgroup_metrics(y_test, y_pred, demo_test, "X2", sex_names)
    edu_df = subgroup_metrics(y_test, y_pred, demo_test, "X3", education_names)
    mar_df = subgroup_metrics(y_test, y_pred, demo_test, "X4", marriage_names)

    print(f"\n--- BY SEX ---")
    print(sex_df.to_string(index=False))

    print(f"\n--- BY EDUCATION ---")
    print(edu_df.to_string(index=False))

    print(f"\n--- BY MARRIAGE ---")
    print(mar_df.to_string(index=False))

    # Disparity check
    print(f"\n--- DISPARITY FLAGS ---")
    for name, df in [("SEX", sex_df), ("EDUCATION", edu_df), ("MARRIAGE", mar_df)]:
        if len(df) > 1:
            recall_range = df["Recall"].max() - df["Recall"].min()
            fpr_range = df["FPR"].max() - df["FPR"].min()
            fnr_range = df["FNR"].max() - df["FNR"].min()

            print(f"{name}:")
            print(
                f"  Recall range: {recall_range:.3f} {'⚠️ HIGH' if recall_range > 0.1 else '✓ OK'}"
            )
            print(
                f"  FPR range:    {fpr_range:.3f} {'⚠️ HIGH' if fpr_range > 0.1 else '✓ OK'}"
            )
            print(
                f"  FNR range:    {fnr_range:.3f} {'⚠️ HIGH' if fnr_range > 0.1 else '✓ OK'}"
            )

    # Save CSVs
    prefix = model_name
    sex_df.to_csv(output_dir / f"{prefix}_sex.csv", index=False)
    edu_df.to_csv(output_dir / f"{prefix}_education.csv", index=False)
    mar_df.to_csv(output_dir / f"{prefix}_marriage.csv", index=False)

    print(f"\nSaved to {output_dir}/")

    return {"sex": sex_df, "education": edu_df, "marriage": mar_df}


# ============================================================
# 4. GENERATE VISUALIZATIONS
# ============================================================


def plot_fairness_results(
    results: dict[str, pd.DataFrame], model_name: str, output_dir: Path
) -> None:
    """
    Create and save fairness visualization plots.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Approval rates
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (demo_name, df) in zip(axes, results.items()):
        sns.barplot(
            data=df,
            x="Group",
            y="Approval_Rate",
            hue="Group",
            ax=ax,
            palette="viridis",
            legend=False,
        )
        ax.set_title(f"Approval Rate by {demo_name.upper()}")
        ax.set_ylim(0, 1)
        for i, row in df.iterrows():
            ax.text(
                i,
                row["Approval_Rate"] + 0.02,
                f"{row['Approval_Rate']:.1%}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    plt.tight_layout()
    plt.savefig(
        output_dir / f"{model_name}_approval_rates.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Plot 2: Recall
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (demo_name, df) in zip(axes, results.items()):
        mean_recall = df["Recall"].mean()
        sns.barplot(
            data=df,
            x="Group",
            y="Recall",
            hue="Group",
            ax=ax,
            palette="coolwarm",
            legend=False,
        )
        ax.axhline(y=mean_recall, color="red", linestyle="--", alpha=0.5)
        ax.set_title(f"Recall by {demo_name.upper()}")
        ax.set_ylim(0, 1)
        for i, row in df.iterrows():
            ax.text(
                i,
                row["Recall"] + 0.02,
                f"{row['Recall']:.1%}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    plt.tight_layout()
    plt.savefig(
        output_dir / f"{model_name}_recall_by_group.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Plot 3: FNR
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (demo_name, df) in zip(axes, results.items()):
        sns.barplot(
            data=df,
            x="Group",
            y="FNR",
            hue="Group",
            ax=ax,
            palette="rocket",
            legend=False,
        )
        ax.set_title(f"False Negative Rate by {demo_name.upper()}")
        ax.set_ylim(0, 1)
        for i, row in df.iterrows():
            ax.text(
                i,
                row["FNR"] + 0.02,
                f"{row['FNR']:.1%}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    plt.tight_layout()
    plt.savefig(
        output_dir / f"{model_name}_false_negative_rates.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Plots saved to {output_dir}/")


# ============================================================
# 5. MAIN
# ============================================================


def main() -> None:
    """Run fairness analysis for key models."""
    models_dir = Path("models")
    output_dir = Path("results/fairness")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_files = [
        "lightgbm_44features.pkl",
        "xgboost_44features.pkl",
        "random_forest_44features.pkl",
    ]

    for model_file in model_files:
        model_path = models_dir / model_file
        if not model_path.exists():
            print(f"Skipping {model_file} (not found)")
            continue

        try:
            results = analyze_model(
                model_path, "data/processed/44features.pkl", output_dir
            )
            plot_fairness_results(results, model_path.stem, output_dir)
        except Exception as e:
            print(f"Error analyzing {model_file}: {e}")
            import traceback

            traceback.print_exc()

    print(f"\n{'='*60}")
    print("FAIRNESS ANALYSIS COMPLETE")
    print(f"Results in: {output_dir.absolute()}")
    print("\nFiles created:")
    for f in sorted(output_dir.glob("*.csv")):
        print(f"  - {f.name}")
    for f in sorted(output_dir.glob("*.png")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
