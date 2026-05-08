"""
SHAP Explainability Module.

Generates global and local explanations for tree-based credit scoring models
using SHapley Additive exPlanations (SHAP). Outputs feature importance rankings
and summary beeswarm plots to visualize directional impacts of financial behavior.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

warnings.filterwarnings("ignore")


def analyze_shap_tree(model_path: Path, data_path: Path, output_dir: Path) -> None:
    """
    Computes SHAP values for a tree-based model and generates interpretability plots.

    Args:
        model_path (Path): Path to the serialized tree-based model.
        data_path (Path): Path to the processed test dataset.
        output_dir (Path): Directory where SHAP visual artifacts will be saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = model_path.stem

    print(f"\nAnalyzing SHAP values for: {model_name}")

    # 1. Load Model and Data
    model = joblib.load(model_path)
    data = joblib.load(data_path)
    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = data[:7]

    # Use a representative sample to speed up computation if the test set is massive
    # SHAP calculates interactions, which can be computationally expensive
    X_sample = shap.sample(X_test, 1000, random_state=42)

    # 2. Initialize TreeExplainer
    # TreeExplainer is highly optimized for XGBoost, LightGBM, and Random Forest
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    except Exception as e:
        print(f"Error computing SHAP values (Model might not be tree-based): {e}")
        return

    # Handle difference in output shapes between RF (list) and XGB/LGBM (array)
    if isinstance(shap_values, list):
        shap_values_pos = shap_values[1]  # Extract values for the 'Default' class
    else:
        shap_values_pos = shap_values

    # 3. Generate Summary Plot (Beeswarm)
    # Shows feature importance AND the directional impact (e.g., does high utilization push risk up?)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values_pos, X_sample, show=False)

    summary_path = output_dir / f"{model_name}_shap_summary.png"
    plt.savefig(summary_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" -> Saved SHAP Summary Plot: {summary_path.name}")

    # 4. Generate Bar Plot (Global Importance)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_pos, X_sample, plot_type="bar", show=False)

    bar_path = output_dir / f"{model_name}_shap_importance_bar.png"
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" -> Saved SHAP Bar Plot: {bar_path.name}")


def main() -> None:
    """Executes the SHAP interpretability pipeline for top-performing tree models."""
    models_dir = Path("models")
    output_dir = Path("results/shap")
    data_path = Path("data/processed/44features.pkl")

    if not data_path.exists():
        print(f"Error: Data file {data_path} not found.")
        return

    # SHAP is optimized for Tree models, so we focus on our gradient boosters and forests
    tree_models = [
        models_dir / "lightgbm_44features.pkl",
        models_dir / "xgboost_44features.pkl",
        models_dir / "random_forest_44features.pkl",
    ]

    print(f"\n{'='*60}")
    print("SHAP EXPLAINABILITY ANALYSIS")
    print(f"{'='*60}")

    for model_path in tree_models:
        if model_path.exists():
            analyze_shap_tree(model_path, data_path, output_dir)
        else:
            print(f"Skipping {model_path.name}: File not found.")


if __name__ == "__main__":
    main()
