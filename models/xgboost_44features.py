# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 17:27:39 2026

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
RAW_FILE = BASE_DIR / "data" / "raw" / "data.xls"
SAVE_DIR = BASE_DIR / "saved_models"
SAVE_DIR.mkdir(exist_ok=True)


# ============================================================
# 1. LOAD RAW DATA
# ============================================================
def load_raw_data(file_path: Path) -> pd.DataFrame:
    # For .xls you may need: pip install xlrd
    df = pd.read_excel(file_path)

    # In this dataset, the first row often contains the real column names
    first_row = df.iloc[0].astype(str).tolist()
    if "LIMIT_BAL" in first_row or "default payment next month" in first_row:
        df.columns = first_row
        df = df.iloc[1:].copy()

    if "default payment next month" in df.columns:
        df = df.rename(columns={"default payment next month": "DEFAULT"})

    if df.columns[0] != "ID":
        df = df.rename(columns={df.columns[0]: "ID"})

    # convert numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["DEFAULT"]).copy()
    df["DEFAULT"] = df["DEFAULT"].astype(int)

    return df


# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    pay_status_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    bill_cols = ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]
    pay_amt_cols = ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]

    eps = 1.0

    # delinquency features
    out["delq_max"] = out[pay_status_cols].max(axis=1)
    out["delq_mean"] = out[pay_status_cols].mean(axis=1)
    out["delq_count_positive"] = (out[pay_status_cols] > 0).sum(axis=1)
    out["delq_count_severe"] = (out[pay_status_cols] >= 2).sum(axis=1)
    out["delq_recent"] = out["PAY_0"]
    out["delq_trend"] = out["PAY_0"] - out["PAY_6"]
    out["ever_severe_delq"] = (out["delq_max"] >= 2).astype(int)

    # bill features
    out["bill_mean"] = out[bill_cols].mean(axis=1)
    out["bill_max"] = out[bill_cols].max(axis=1)
    out["bill_std"] = out[bill_cols].std(axis=1).fillna(0)
    out["bill_trend"] = out["BILL_AMT1"] - out["BILL_AMT6"]

    # payment features
    out["pay_mean"] = out[pay_amt_cols].mean(axis=1)
    out["pay_max"] = out[pay_amt_cols].max(axis=1)
    out["pay_std"] = out[pay_amt_cols].std(axis=1).fillna(0)
    out["pay_trend"] = out["PAY_AMT1"] - out["PAY_AMT6"]
    out["zero_pay_count"] = (out[pay_amt_cols] == 0).sum(axis=1)

    util_cols = []
    ratio_cols = []

    for i in range(1, 7):
        util_col = f"util_{i}"
        ratio_col = f"pay_ratio_{i}"

        out[util_col] = out[f"BILL_AMT{i}"] / (out["LIMIT_BAL"] + eps)
        out[ratio_col] = out[f"PAY_AMT{i}"] / (out[f"BILL_AMT{i}"].abs() + eps)

        util_cols.append(util_col)
        ratio_cols.append(ratio_col)

    out["bill_utilization_mean"] = out[util_cols].mean(axis=1)
    out["bill_utilization_max"] = out[util_cols].max(axis=1)
    out["high_util_count"] = (out[util_cols] > 0.8).sum(axis=1)

    out["pay_ratio_mean"] = out[ratio_cols].mean(axis=1)
    out["pay_ratio_min"] = out[ratio_cols].min(axis=1)
    out["underpay_count"] = (out[ratio_cols] < 0.2).sum(axis=1)

    out["avg_bill_minus_pay"] = out["bill_mean"] - out["pay_mean"]
    out["recent_bill_minus_pay"] = out["BILL_AMT1"] - out["PAY_AMT1"]

    return out


# ============================================================
# 3. CHOOSE FEATURE SET
# ============================================================
def get_feature_columns():
    # raw behavioral
    raw_behavioral = [
        "LIMIT_BAL", "AGE",
        "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
    ]

    # engineered features
    engineered = [
        "delq_max",
        "delq_mean",
        "delq_count_positive",
        "delq_count_severe",
        "delq_recent",
        "delq_trend",
        "ever_severe_delq",
        "bill_mean",
        "bill_max",
        "bill_std",
        "bill_trend",
        "pay_mean",
        "pay_max",
        "pay_std",
        "pay_trend",
        "zero_pay_count",
        "bill_utilization_mean",
        "bill_utilization_max",
        "high_util_count",
        "pay_ratio_mean",
        "pay_ratio_min",
        "underpay_count",
        "avg_bill_minus_pay",
        "recent_bill_minus_pay"
    ]

    # final chosen set
    feature_cols = raw_behavioral + engineered
    return feature_cols


# ============================================================
# 4. TRAIN / VAL / TEST SPLIT
# ============================================================
def make_splits(df: pd.DataFrame, feature_cols: list):
    X = df[feature_cols].copy()
    y = df["DEFAULT"].astype(int).copy()

    # 70% train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.30,
        stratify=y,
        random_state=42
    )

    # 15% val, 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


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
    df_raw = load_raw_data(RAW_FILE)
    df = engineer_features(df_raw)
    feature_cols = get_feature_columns()

    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(df, feature_cols)

    print("Default rate:")
    print("Train:", y_train.mean())
    print("Val:  ", y_val.mean())
    print("Test: ", y_test.mean())

    print("\nNumber of features:", len(feature_cols))
    print("\nColumns:", feature_cols)

    xgb_model = train_xgboost(X_train, y_train, X_val, y_val)
    
    joblib.dump(xgb_model, "xgboost_44features.pkl")
    print(f"Model saved to {'xgboost_44features.pkl'}")
    
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
    plt.title("Top 20 XGBoost Feature Importances")
    plt.grid(True)
    plt.show()

    # Save model
    joblib.dump(xgb_model, SAVE_DIR / "xgb_engineered_features.pkl")


if __name__ == "__main__":
    main()
