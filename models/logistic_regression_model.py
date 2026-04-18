import joblib
import matplotlib.pyplot as plt

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
# 1. Load the Pre-Split Enhanced Dataset
# ---------------------------------------------------------
file_path = "44features.pkl" 
X_train, X_val, X_test, y_train, y_val, y_test = joblib.load(file_path)

# ---------------------------------------------------------
# 2. Preprocess and Train Logistic Regression
# ---------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) 

lr_clf = LogisticRegression(max_iter=1000, random_state=42)
lr_clf.fit(X_train_scaled, y_train)

# Generate Predictions and Probabilities
y_pred_lr = lr_clf.predict(X_test_scaled)
y_proba_lr = lr_clf.predict_proba(X_test_scaled)[:, 1]

# ---------------------------------------------------------
# 3. Print Metrics
# ---------------------------------------------------------
print("=== Logistic Regression Baseline ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"ROC AUC:  {roc_auc_score(y_test, y_proba_lr):.4f}")
print(f"PR AUC:   {average_precision_score(y_test, y_proba_lr):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr, zero_division=0))

# ---------------------------------------------------------
# 4. Generate Graphs
# ---------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 11)) # Slightly adjusted height

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
