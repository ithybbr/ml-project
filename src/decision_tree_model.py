from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier


DEFAULT_RANDOM_STATE = 42

DEFAULT_DATASET_FILENAME = "18features.pkl"

DEFAULT_PARAM_GRID = {
    "criterion": ["gini", "entropy"],
    "max_depth": [3, 4, 5, 6, 8, 10, None],
    "min_samples_split": [2, 10, 25, 50],
    "min_samples_leaf": [10, 20, 50, 100],
    "class_weight": [None, "balanced"],
    "ccp_alpha": [0.0, 0.001, 0.005, 0.01],
}


@dataclass
class DecisionTreeConfig:
    dataset_filename: str = DEFAULT_DATASET_FILENAME
    output_filename: str = "decision_tree_model.pkl"
    scoring: str = "average_precision"  # PR AUC
    inner_splits: int = 3
    outer_splits: int = 5
    random_state: int = DEFAULT_RANDOM_STATE


def find_repo_root(start_path: Path | None = None) -> Path:
    if start_path is None:
        start_path = Path.cwd()

    current = start_path.resolve()
    candidates = [current, *current.parents]

    for candidate in candidates:
        if (candidate / "models").exists() and (candidate / "src").exists():
            return candidate

    raise FileNotFoundError(
        "Could not find repository root. Run this script from inside the project repository."
    )


def resolve_dataset_path(repo_root: Path, dataset_filename: str) -> Path:
    dataset_path = repo_root / "models" / dataset_filename

    if not dataset_path.exists():
        available_pkls = sorted(path.name for path in (repo_root / "models").glob("*.pkl"))
        raise FileNotFoundError(
            f"Could not find {dataset_path}. Available .pkl files in models/: {available_pkls}"
        )

    return dataset_path


def _to_dataframe(X: Any) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.reset_index(drop=True)
    return pd.DataFrame(X).reset_index(drop=True)


def _to_series(y: Any) -> pd.Series:
    if isinstance(y, pd.Series):
        return y.astype(int).reset_index(drop=True)
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError("Target DataFrame must contain exactly one column.")
        return y.iloc[:, 0].astype(int).reset_index(drop=True)
    return pd.Series(y).astype(int).reset_index(drop=True)


def load_dataset_from_pkl(dataset_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    joblib.load(dataset_path)

    if not isinstance(data, (tuple, list)):
        raise TypeError(
            "Expected the dataset .pkl file to contain a tuple/list with train, validation, and test data."
        )

    if len(data) < 6:
        raise ValueError(
            "Expected at least 6 objects: X_train, X_val, X_test, y_train, y_val, y_test."
        )

    X_train, X_val, X_test, y_train, y_val, y_test = data[:6]

    return (
        _to_dataframe(X_train),
        _to_dataframe(X_val),
        _to_dataframe(X_test),
        _to_series(y_train),
        _to_series(y_val),
        _to_series(y_test),
    )


def make_nested_cv_search(config: DecisionTreeConfig) -> tuple[GridSearchCV, StratifiedKFold]:
    base_model = DecisionTreeClassifier(random_state=config.random_state)

    inner_cv = StratifiedKFold(
        n_splits=config.inner_splits,
        shuffle=True,
        random_state=config.random_state,
    )

    outer_cv = StratifiedKFold(
        n_splits=config.outer_splits,
        shuffle=True,
        random_state=config.random_state,
    )

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=DEFAULT_PARAM_GRID,
        scoring=config.scoring,
        cv=inner_cv,
        n_jobs=-1,
        refit=True,
    )

    return grid_search, outer_cv


def train_decision_tree_with_nested_cv(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    config: DecisionTreeConfig,
) -> tuple[DecisionTreeClassifier, dict[str, Any]]:

    X_train_val = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_train_val = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)

    grid_search, outer_cv = make_nested_cv_search(config)

    nested_cv_scores = cross_val_score(
        grid_search,
        X_train_val,
        y_train_val,
        cv=outer_cv,
        scoring=config.scoring,
        n_jobs=-1,
    )

    final_search = grid_search.fit(X_train_val, y_train_val)
    final_model = final_search.best_estimator_

    metadata = {
        "model_name": "DecisionTreeClassifier",
        "dataset_filename": config.dataset_filename,
        "scoring": config.scoring,
        "inner_splits": config.inner_splits,
        "outer_splits": config.outer_splits,
        "nested_cv_scores": nested_cv_scores,
        "nested_cv_mean": float(np.mean(nested_cv_scores)),
        "nested_cv_std": float(np.std(nested_cv_scores)),
        "best_params": final_search.best_params_,
        "best_inner_cv_score": float(final_search.best_score_),
        "n_features": int(final_model.n_features_in_),
    }

    return final_model, metadata


def save_model(model: DecisionTreeClassifier, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)


def save_metadata(metadata: dict[str, Any], output_path: Path) -> None:
    metadata_path = output_path.with_suffix(".metadata.pkl")
    joblib.dump(metadata, metadata_path)


def train_and_save_from_repo(
    repo_root: Path,
    config: DecisionTreeConfig,
    output_path: Path | None = None,
) -> tuple[DecisionTreeClassifier, dict[str, Any], Path]:
    dataset_path = resolve_dataset_path(repo_root, config.dataset_filename)
    X_train, X_val, _X_test, y_train, y_val, _y_test = load_dataset_from_pkl(dataset_path)

    model, metadata = train_decision_tree_with_nested_cv(
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        config=config,
    )

    if output_path is None:
        output_path = repo_root / "models" / config.output_filename

    save_model(model, output_path)
    save_metadata(metadata, output_path)

    return model, metadata, output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Decision Tree using nested cross-validation and save the final model."
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default=None,
        help="Path to repository root. If omitted, it is found automatically.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET_FILENAME,
        help="Dataset .pkl filename inside models/. Default: 18features.pkl",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output model path. Default: <repo_root>/models/decision_tree_model.pkl",
    )
    parser.add_argument(
        "--scoring",
        type=str,
        default="average_precision",
        help="GridSearchCV scoring. Default: average_precision (PR AUC).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = Path(args.repo_root).resolve() if args.repo_root else find_repo_root()
    output_path = Path(args.output).resolve() if args.output else None

    config = DecisionTreeConfig(
        dataset_filename=args.dataset,
        scoring=args.scoring,
    )

    model, metadata, saved_path = train_and_save_from_repo(
        repo_root=repo_root,
        config=config,
        output_path=output_path,
    )

    print(f"Saved model to: {saved_path}")
    print(f"Saved metadata to: {saved_path.with_suffix('.metadata.pkl')}")
    print(f"Dataset used: {metadata['dataset_filename']}")
    print(f"Best params: {metadata['best_params']}")
    print(
        f"Nested CV {metadata['scoring']} mean ± std: "
        f"{metadata['nested_cv_mean']:.4f} ± {metadata['nested_cv_std']:.4f}"
    )
    print(f"Number of features: {metadata['n_features']}")


if __name__ == "__main__":
    main()
