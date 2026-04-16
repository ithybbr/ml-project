# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 15:16:29 2026

@author: User
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.linear_model import LogisticRegression
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


# =========================
# 1. Load data
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

print("BASE_DIR =", BASE_DIR)
print("DATA_DIR =", DATA_DIR)
print("X_train exists:", (DATA_DIR / "X_train.csv").exists())

X_train = pd.read_csv(DATA_DIR / "X_train.csv")
X_val   = pd.read_csv(DATA_DIR / "X_val.csv")
X_test  = pd.read_csv(DATA_DIR / "X_test.csv")

y_train = pd.read_csv(DATA_DIR / "y_train.csv").iloc[:, 0].astype(int)
y_val   = pd.read_csv(DATA_DIR / "y_val.csv").iloc[:, 0].astype(int)
y_test  = pd.read_csv(DATA_DIR / "y_test.csv").iloc[:, 0].astype(int) 

print("\nDefault rate:")
print("Train:", y_train.mean())
print("Val:  ", y_val.mean())
print("Test: ", y_test.mean())

print("\nColumns:", X_train.columns.tolist())


# =========================
# 2. Helper functions
# =========================
def find_best_threshold(y_true, y_proba):
    """
    Finds threshold that maximizes F1 score on validation set.
    """
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
    """
    Returns standard classification and probability metrics.
    """
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


# =========================
# 3. Logistic Regression baseline
# =========================
log_reg = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)

log_reg.fit(X_train, y_train)

val_results_lr, val_proba_lr, val_pred_lr = evaluate_model(
    "Logistic Regression", log_reg, X_val, y_val, threshold=0.5
)

best_thr_lr, best_f1_lr = find_best_threshold(y_val, val_proba_lr)
print(f"\nBest validation threshold for Logistic Regression: {best_thr_lr:.2f}, F1={best_f1_lr:.4f}")

test_results_lr, test_proba_lr, test_pred_lr = evaluate_model(
    "Logistic Regression", log_reg, X_test, y_test, threshold=best_thr_lr
)

print_results("Validation results - Logistic Regression", val_results_lr)
print_results("Test results - Logistic Regression", test_results_lr)


# =========================
# 4. XGBoost model
# =========================
# Since default class is minority, this helps the model pay more attention to it.
scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

xgb_model = XGBClassifier(
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

xgb_model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

val_results_xgb, val_proba_xgb, val_pred_xgb = evaluate_model(
    "XGBoost", xgb_model, X_val, y_val, threshold=0.5
)

best_thr_xgb, best_f1_xgb = find_best_threshold(y_val, val_proba_xgb)
print(f"\nBest validation threshold for XGBoost: {best_thr_xgb:.2f}, F1={best_f1_xgb:.4f}")

test_results_xgb, test_proba_xgb, test_pred_xgb = evaluate_model(
    "XGBoost", xgb_model, X_test, y_test, threshold=best_thr_xgb
)

print_results("Validation results - XGBoost", val_results_xgb)
print_results("Test results - XGBoost", test_results_xgb)


# =========================
# 5. Compare models
# =========================
comparison = pd.DataFrame([
    test_results_lr,
    test_results_xgb
])

print("\nModel comparison on TEST set:")
print(comparison[[
    "model", "threshold", "roc_auc", "pr_auc",
    "log_loss", "brier_score", "accuracy",
    "precision", "recall", "f1"
]].sort_values(by="roc_auc", ascending=False))


# =========================
# 6. Plots for XGBoost
# =========================
plot_roc_pr_curves(y_test, test_proba_xgb, "XGBoost")
plot_confusion(y_test, test_pred_xgb, "XGBoost")
plot_calibration(y_test, test_proba_xgb, "XGBoost")


# =========================
# 7. Feature importance
# =========================
importances = pd.Series(
    xgb_model.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

print("\nXGBoost feature importances:")
print(importances)

plt.figure(figsize=(6, 4))
plt.bar(importances.index, importances.values)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("XGBoost Feature Importance")
plt.grid(True)
plt.show()