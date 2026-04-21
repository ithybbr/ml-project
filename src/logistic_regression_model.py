import joblib
import numpy as np
import matplotlib.pyplot as plt
import os 

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, 
    roc_auc_score, 
    accuracy_score, 
    average_precision_score, 
    RocCurveDisplay, 
    PrecisionRecallDisplay, 
    ConfusionMatrixDisplay
)
from sklearn.calibration import CalibrationDisplay

# ---------------------------------------------------------
# 1. Load the Data (Behavioral Variables Only - 18 Features)
# ---------------------------------------------------------
file_path = "models/3features.pkl"       #"18features"
print(f"Loading data from {file_path}...")

# Unpacking all 7 items exactly as they are saved in your .pkl file
X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = joblib.load(file_path)

# Force the target variables to be discrete integers (0 and 1)
y_train = y_train.astype(int)
y_val = y_val.astype(int)
y_test = y_test.astype(int)

print(f"Dataset Loaded Successfully!")
print(f"Behavioral features in training set: {X_train.shape[1]}\n")

# ---------------------------------------------------------
# 2. Preprocess and Train Logistic Regression
# ---------------------------------------------------------


lr_clf = LogisticRegression(max_iter=1000, random_state=42)
lr_clf.fit(X_train, y_train)

# Generate Predictions and Probabilities
y_pred_lr = lr_clf.predict(X_test)
y_proba_lr = lr_clf.predict_proba(X_test)[:, 1]

# ---------------------------------------------------------
# 3. Print Metrics
# ---------------------------------------------------------
print("=== Logistic Regression (18 Behavioral Variables) ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"ROC AUC:  {roc_auc_score(y_test, y_proba_lr):.4f}")
print(f"PR AUC:   {average_precision_score(y_test, y_proba_lr):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr, zero_division=0))

# ---------------------------------------------------------
# 4. Generate Graphs
# ---------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# 1. Confusion Matrix
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_lr, ax=axes[0, 0], cmap='Blues', colorbar=False
)
axes[0, 0].set_title("Confusion Matrix", fontsize=15, pad=15)

# 2. ROC Curve
RocCurveDisplay.from_predictions(
    y_test, y_proba_lr, name="Logistic Regression", ax=axes[0, 1]
)
axes[0, 1].set_title("ROC AUC Curve", fontsize=15, pad=15)
axes[0, 1].grid(True, alpha=0.3)

# 3. Precision-Recall (PR) Curve
PrecisionRecallDisplay.from_predictions(
    y_test, y_proba_lr, name="Logistic Regression", ax=axes[1, 0]
)
axes[1, 0].set_title("Precision-Recall (PR) Curve", fontsize=15, pad=15)
axes[1, 0].grid(True, alpha=0.3)

# 4. Calibration Curve
CalibrationDisplay.from_predictions(
    y_test, y_proba_lr, name="Logistic Regression", ax=axes[1, 1], n_bins=10
)
axes[1, 1].set_title("Calibration Curve (Reliability)", fontsize=15, pad=15)
axes[1, 1].grid(True, alpha=0.3)

# FIX: Add specific height (h_pad) and width (w_pad) spacing between the subplots
plt.tight_layout(h_pad=4.0, w_pad=3.0)
plt.show()

# ---------------------------------------------------------
# 6. SAVE THE TRAINED MODEL AND SCALER
# ---------------------------------------------------------
output_path = "models/logreg_3features.pkl"

# Ensure the models directory exists just in case
os.makedirs("models", exist_ok=True)

# Export the pickle file
joblib.dump(lr_clf, output_path)

print(f"\n[SUCCESS] Model successfully saved to: {output_path}")