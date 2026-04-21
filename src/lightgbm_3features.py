import joblib
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss

# ---------------------------------------------------------
# 1. LOAD THE DATA FROM .PKL
# ---------------------------------------------------------
print("Loading data from 3features.pkl...")

X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = joblib.load("models/3features.pkl")

y_train = y_train.astype(int)
y_val = y_val.astype(int)
y_test = y_test.astype(int)

# ---------------------------------------------------------
# 2. INITIALIZE AND TRAIN LIGHTGBM
# ---------------------------------------------------------
print("Training the LightGBM model (this is fast)...")

# With only 3 features we use a smaller num_leaves to avoid overfitting
# n_estimators=500: number of boosting rounds (trees)
# learning_rate=0.05: step size shrinkage
# num_leaves=15: smaller leaf count suits a low-dimensional problem
# class_weight='balanced': compensates for the imbalanced default rate
# n_jobs=-1: use all CPU cores
# random_state=42: ensures reproducibility
lgbm_model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=15,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=1.0,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

lgbm_model.fit(X_train, y_train)

# ---------------------------------------------------------
# 3. EVALUATE THE MODEL
# ---------------------------------------------------------
print("Evaluating the model on Validation data...")

y_val_probs = lgbm_model.predict_proba(X_val)[:, 1]

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
model_path = 'models/lightgbm_3features.pkl'
joblib.dump(lgbm_model, model_path)

print(f"Success! Model saved to {model_path}")
