from __future__ import annotations

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

warnings.filterwarnings("ignore")

# --- Configuration ---
RANDOM_STATE = 42
SCORING_METRIC = "average_precision"

# Hyperparameter grid for Nested CV
# Kept concise because Scikit-Learn's Gradient Boosting is strictly single-threaded 
# during tree building and can take longer than XGBoost or LightGBM.
DEFAULT_PARAM_GRID = {
    "n_estimators": [100, 200],
    "learning_rate": [0.05, 0.1],
    "max_depth": [3, 4, 5],
    "subsample": [0.8, 1.0],
    "min_samples_split": [2, 10],
}


def load_and_prepare_data(dataset_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Loads the dataset and combines train/val sets for cross-validation."""
    data = joblib.load(dataset_path)
    
    # Extract the first 6 elements: X_train, X_val, X_test, y_train, y_val, y_test
    X_train, X_val, _, y_train, y_val, _ = data[:6]

    # Convert to pandas objects to ensure safe concatenation
    X_train_df = pd.DataFrame(X_train)
    X_val_df = pd.DataFrame(X_val)
    
    # Handle single-column dataframes or series for y
    y_train_series = pd.Series(np.ravel(y_train)).astype(int)
    y_val_series = pd.Series(np.ravel(y_val)).astype(int)

    # Combine training and validation sets
    X_combined = pd.concat([X_train_df, X_val_df], axis=0).reset_index(drop=True)
    y_combined = pd.concat([y_train_series, y_val_series], axis=0).reset_index(drop=True)

    return X_combined, y_combined


def train_with_nested_cv(X: pd.DataFrame, y: pd.Series) -> GradientBoostingClassifier:
    """Performs nested CV for evaluation and trains the final model."""
    
    base_model = GradientBoostingClassifier(random_state=RANDOM_STATE)

    # Configure inner and outer cross-validation strategies
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=DEFAULT_PARAM_GRID,
        scoring=SCORING_METRIC,
        cv=inner_cv,
        n_jobs=-1,  # Parallelize across folds and parameter combinations
        refit=True, # Ensures the final model is trained on the full dataset passed to fit()
    )

    # 1. Nested Cross-Validation (Evaluation Step)
    print("Running nested cross-validation...")
    nested_cv_scores = cross_val_score(
        grid_search, X, y, cv=outer_cv, scoring=SCORING_METRIC, n_jobs=-1
    )
    
    mean_score = np.mean(nested_cv_scores)
    std_score = np.std(nested_cv_scores)
    print(f"Nested CV {SCORING_METRIC}: {mean_score:.4f} ± {std_score:.4f}")

    # 2. Final Model Training (Fit Step)
    print("Tuning hyperparameters and fitting the final model...")
    grid_search.fit(X, y)
    
    print(f"Best hyperparameters found: {grid_search.best_params_}")
    return grid_search.best_estimator_


def main() -> None:
    n_features = [3, 18, 44]
    
    for n in n_features:
        print("\n" + "="*50)
        print(f"PROCESSING DATASET: {n} FEATURES")
        print("="*50)
        
        # Adjust paths based on your directory structure
        dataset_path = Path(f"../data/processed/{n}features.pkl")
        output_model_path = Path(f"../models/gradient_boosting_{n}features.pkl")

        # Fallback to local path if running from root
        if not dataset_path.exists():
            dataset_path = Path(f"data/processed/{n}features.pkl")
            output_model_path = Path(f"models/gradient_boosting_{n}features.pkl")
            if not dataset_path.exists():
                print(f"Dataset not found at {dataset_path}. Skipping.")
                continue

        # Execute pipeline
        print(f"Loading data from {dataset_path}...")
        X, y = load_and_prepare_data(dataset_path)
        
        print(f"Combined Default rate: {y.mean():.4f}")

        final_model = train_with_nested_cv(X, y)

        # Save the final model
        output_model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_model, output_model_path)
        print(f"Model successfully saved to {output_model_path}")


if __name__ == "__main__":
    main()