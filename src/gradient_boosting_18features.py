import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss

# ---------------------------------------------------------
# 1. LOAD THE DATA FROM .PKL
# ---------------------------------------------------------
print("Loading data from 18features.pkl...")

X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = joblib.load("models/18features.pkl")

y_train = y_train.astype(int)
y_val = y_val.astype(int)
y_test = y_test.astype(int)

# ---------------------------------------------------------
# 2. INITIALIZE AND TRAIN GRADIENT BOOSTING
# ---------------------------------------------------------
print("Training the Gradient Boosting model (this can take a minute)...")

# n_estimators=200: number of sequential boosting stages (trees)
# learning_rate=0.1: how strongly each new tree corrects previous mistakes
# max_depth=3: shallow trees, typical for boosting (weak learners)
# random_state=42: ensures reproducibility
gb_model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

gb_model.fit(X_train, y_train)

# ---------------------------------------------------------
# 3. EVALUATE THE MODEL
# ---------------------------------------------------------
print("Evaluating the model on Validation data...")

y_val_probs = gb_model.predict_proba(X_val)[:, 1]

roc_auc = roc_auc_score(y_val, y_val_probs)
pr_auc = average_precision_score(y_val, y_val_probs)
logloss = log_loss(y_val, y_val_probs)
brier = brier_score_loss(y_val, y_val_probs)

print("\n--- Validation Results ---")
print(f"ROC-AUC:      {roc_auc:.4f}")
print(f"PR-AUC:       {pr_auc:.4f}")
print(f"Log Loss:     {logloss:.4f}")
print(f"Brier Score:  {brier:.4f}")
print("--------------------------\n")

# ---------------------------------------------------------
# 4. SAVE THE TRAINED MODEL
# ---------------------------------------------------------
model_path = 'models/gradient_boosting_18features.pkl'
joblib.dump(gb_model, model_path)

print(f"Success! Model saved to {model_path}")
