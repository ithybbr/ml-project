import joblib
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------------
# 1. LOAD THE DATA FROM .PKL
# ---------------------------------------------------------
print("Loading data from 18features.pkl...")

X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = joblib.load("models/18features.pkl")

y_train = y_train.astype(int)
y_val = y_val.astype(int)
y_test = y_test.astype(int)

# ---------------------------------------------------------
# 2. INITIALIZE AND TRAIN THE RANDOM FOREST
# ---------------------------------------------------------
print("Training the Random Forest model (this might take a few seconds)...")

# n_estimators=100: builds 100 decision trees
# n_jobs=-1: uses all CPU cores to train faster
# random_state=42: ensures you get the exact same results every time
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# Train the model using the training data
rf_model.fit(X_train, y_train)

# ---------------------------------------------------------
# 4. SAVE THE TRAINED MODEL
# ---------------------------------------------------------
# Save the trained model so it can be used later without retraining
model_path = 'models/random_forest_18features.pkl'
joblib.dump(rf_model, model_path)

print(f"Success! Model saved to {model_path}")
