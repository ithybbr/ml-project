import joblib
import numpy as np
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
# 1. Load the Data (Behavioral Variables Only - 18 Features)
# ---------------------------------------------------------
file_path = "models/18features.pkl"       #"18features"
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
# 5. Prediction Function For New Data
# ---------------------------------------------------------
def predict_new_client_behavior(model, fitted_scaler, new_behavioral_data):
    """
    Takes in raw behavioral data for new clients, scales it, and returns predictions.
    
    Parameters:
    - model: The trained Logistic Regression model
    - fitted_scaler: The StandardScaler fitted on the training data
    - new_behavioral_data: A list, numpy array, or 2D array of behavioral features
                           (Must have exactly 18 features to match the model)
                           
    Returns:
    - predictions (0 or 1)
    - probabilities (float between 0 and 1 representing default risk)
    """
    new_data_np = np.array(new_behavioral_data)
    
    # If a single row/client is passed (1D array), reshape it to 2D matrix
    if new_data_np.ndim == 1:
        new_data_np = new_data_np.reshape(1, -1)
        
    # Ensure the new data has exactly 18 features
    expected_features = fitted_scaler.n_features_in_
    if new_data_np.shape[1] != expected_features:
        raise ValueError(f"Expected {expected_features} behavioral features, got {new_data_np.shape[1]}")
        
    # Scale the new data using the EXACT SAME scaler fitted on the training data
    new_data_scaled = fitted_scaler.transform(new_data_np)
    
    # Make predictions
    preds = model.predict(new_data_scaled)
    probs = model.predict_proba(new_data_scaled)[:, 1]
    
    return preds, probs

# --- Example of how to use the function ---
print("\n--- Testing Prediction Function on 'New' Data ---")
# Let's pretend the first 3 rows of the test set are entirely new clients passing through our system
fake_new_clients = X_test[:3]

preds, probs = predict_new_client_behavior(lr_clf, scaler, fake_new_clients)

for i in range(len(preds)):
    print(f"Client {i+1} -> Predicted Class: {preds[i]}, Probability of Default: {probs[i]:.2%}")