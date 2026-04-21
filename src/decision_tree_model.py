from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import FixedThresholdClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier


DEFAULT_THRESHOLD = 0.5
DEFAULT_RANDOM_STATE = 42


@dataclass
class DecisionTreeConfig:
    max_depth: int | None = 6
    min_samples_leaf: int = 50
    class_weight: str | dict[str, float] | None = "balanced"
    threshold: float = DEFAULT_THRESHOLD
    random_state: int = DEFAULT_RANDOM_STATE


def find_repo_root(start_path: Path | None = None) -> Path:
    """
    Find the project root by locating data/processed.
    Works whether this script is run from the repo root, src/, or another subfolder.
    """
    if start_path is None:
        start_path = Path.cwd()

    current = start_path.resolve()
    candidates = [current, *current.parents]
    for candidate in candidates:
        if (candidate / "data" / "processed").exists():
            return candidate

    raise FileNotFoundError(
        "Could not find project root containing data/processed. "
        "Run this script from inside the repository."
    )


def load_processed_data(
    repo_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    data_dir = repo_root / "data" / "processed"

    X_train = pd.read_csv(data_dir / "X_train.csv")
    X_val = pd.read_csv(data_dir / "X_val.csv")
    X_test = pd.read_csv(data_dir / "X_test.csv")

    y_train = pd.read_csv(data_dir / "y_train.csv").iloc[:, 0]
    y_val = pd.read_csv(data_dir / "y_val.csv").iloc[:, 0]
    y_test = pd.read_csv(data_dir / "y_test.csv").iloc[:, 0]

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_tree(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    config: DecisionTreeConfig,
) -> FixedThresholdClassifier:
    X_train_full = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_train_full = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)

    base_model = DecisionTreeClassifier(
        random_state=config.random_state,
        max_depth=config.max_depth,
        min_samples_leaf=config.min_samples_leaf,
        class_weight=config.class_weight,
    )
    base_model.fit(X_train_full, y_train_full)

    return FixedThresholdClassifier(estimator=base_model, threshold=config.threshold)


def evaluate_model(
    model: FixedThresholdClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "pr_auc": average_precision_score(y_test, probabilities),
        "confusion_matrix": confusion_matrix(y_test, predictions),
    }


def save_model(model: FixedThresholdClassifier, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)


def train_and_save_from_repo(
    repo_root: Path,
    output_path: Path | None = None,
    config: DecisionTreeConfig | None = None,
) -> tuple[FixedThresholdClassifier, dict[str, Any], Path]:
    if config is None:
        config = DecisionTreeConfig()

    X_train, X_val, X_test, y_train, y_val, y_test = load_processed_data(repo_root)
    model = train_tree(X_train, X_val, y_train, y_val, config)
    metrics = evaluate_model(model, X_test, y_test)

    if output_path is None:
        output_path = repo_root / "models" / "decision_tree.pkl"

    save_model(model, output_path)
    return model, metrics, output_path


def pretty_print_metrics(metrics: dict[str, Any]) -> None:
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"ROC AUC:  {metrics['roc_auc']:.4f}")
    print(f"PR AUC:   {metrics['pr_auc']:.4f}")
    print("Confusion matrix:")
    print(metrics["confusion_matrix"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Decision Tree model and save it as a .pkl file."
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default=None,
        help="Path to the repository root. If omitted, the script will try to find it automatically.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path for the saved .pkl model. Default: <repo_root>/models/decision_tree.pkl",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Decision Tree max_depth.",
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=50,
        help="Decision Tree min_samples_leaf.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Classification threshold applied during evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = Path(args.repo_root).resolve() if args.repo_root else find_repo_root()
    output_path = Path(args.output).resolve() if args.output else repo_root / "models" / "decision_tree_model.pkl"

    config = DecisionTreeConfig(
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        threshold=args.threshold,
    )

    model, metrics, saved_path = train_and_save_from_repo(
        repo_root=repo_root,
        output_path=output_path,
        config=config,
    )

    print("Saved model to:", saved_path)
    print(f"Evaluation threshold: {model.threshold}")
    pretty_print_metrics(metrics)


if __name__ == "__main__":
    main()