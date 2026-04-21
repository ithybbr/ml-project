# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 20:43:10 2026

@author: User
"""


from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve
)
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
BEH_PKL = BASE_DIR / "models" / "behavioral.pkl"
SAVE_DIR = BASE_DIR / "models"
SAVE_DIR.mkdir(exist_ok=True)


# ============================================================
# 1. LOAD RAW DATA
# ============================================================
def load_raw_data(file_path: Path) -> pd.DataFrame:
    df = joblib.load(file_path)
    return df

# ============================================================
# 5. EVALUATION HELPERS
# ============================================================
def find_best_threshold(y_true, y_proba):
    thresholds = np.linspace(0.05, 0.95, 91)
    best_threshold = 0.5
    best_f1 = -1.0

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = t

    return best_threshold, best_f1


def evaluate_model(name, model, X, y, threshold=0.5):
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    results = {
        "model": name,
        "threshold": threshold,
        "roc_auc": roc_auc_score(y, y_proba),
        "pr_auc": average_precision_score(y, y_proba),
        "log_loss": log_loss(y, y_proba),
        "brier_score": brier_score_loss(y, y_proba),
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
    }

    return results, y_proba, y_pred


def print_results(title, results_dict):
    print(f"\n{title}")
    for k, v in results_dict.items():
        if isinstance(v, float):
            print(f"{k:12s}: {v:.4f}")
        else:
            print(f"{k:12s}: {v}")


def plot_roc_pr_curves(y_true, y_proba, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {model_name}")
    plt.grid(True)
    plt.show()


def plot_confusion(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()


def plot_calibration(y_true, y_proba, model_name):
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)

    plt.figure(figsize=(6, 4))
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(f"Calibration Curve - {model_name}")
    plt.grid(True)
    plt.show()


# ============================================================
# 6. TRAIN XGBOOST
# ============================================================
def train_xgboost(X_train, y_train, X_val, y_val):
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric=["auc", "aucpr", "logloss"],
        n_estimators=500,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.2,
        reg_lambda=2.0,
        scale_pos_weight=float(scale_pos_weight),
        random_state=42,
        tree_method="hist",
        early_stopping_rounds=30
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    return model


# ============================================================
# 7. MAIN
# ============================================================
def main():
    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = load_raw_data(BEH_PKL)
    y_train = y_train.astype(int)
    y_val = y_val.astype(int)
    y_test = y_test.astype(int)
    print("Default rate:")
    print("Train:", y_train.mean())
    print("Val:  ", y_val.mean())
    print("Test: ", y_test.mean())

    xgb_model = train_xgboost(X_train, y_train, X_val, y_val)
    
    joblib.dump(xgb_model, SAVE_DIR / "xgboost_18features.pkl")
    print(f"Model saved to {SAVE_DIR / 'xgboost_18features.pkl'}")
    val_results_xgb, val_proba_xgb, val_pred_xgb = evaluate_model(
        "XGBoost (engineered features)", xgb_model, X_val, y_val, threshold=0.5
    )

    best_thr_xgb, best_f1_xgb = find_best_threshold(y_val, val_proba_xgb)
    print(f"\nBest validation threshold for XGBoost: {best_thr_xgb:.2f}, F1={best_f1_xgb:.4f}")

    test_results_xgb, test_proba_xgb, test_pred_xgb = evaluate_model(
        "XGBoost (engineered features)", xgb_model, X_test, y_test, threshold=best_thr_xgb
    )

    print_results("Validation results - XGBoost", val_results_xgb)
    print_results("Test results - XGBoost", test_results_xgb)

    # Plots
    plot_roc_pr_curves(y_test, test_proba_xgb, "XGBoost (engineered features)")
    plot_confusion(y_test, test_pred_xgb, "XGBoost (engineered features)")
    plot_calibration(y_test, test_proba_xgb, "XGBoost (engineered features)")

    # Feature importance
    importances = pd.Series(
        xgb_model.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=False)

    print("\nXGBoost feature importances:")
    print(importances.head(20))

    plt.figure(figsize=(10, 6))
    importances.head(20).sort_values().plot(kind="barh")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Top XGBoost Feature Importances")
    plt.grid(True)
    plt.show()

    # Save model
    joblib.dump(xgb_model, SAVE_DIR / "xgb_engineered_features.pkl")


if __name__ == "__main__":
    main()