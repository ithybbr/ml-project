import joblib
import os 
from sklearn.linear_model import LogisticRegression

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

# ---------------------------------------------------------
# 2. Train Logistic Regression
# ---------------------------------------------------------

lr_clf = LogisticRegression(max_iter=1000, random_state=42)
lr_clf.fit(X_train, y_train)

# ---------------------------------------------------------
# 3. SAVE THE TRAINED MODEL AND SCALER
# ---------------------------------------------------------
output_path = "models/logreg_3features.pkl"

# Ensure the models directory exists just in case
os.makedirs("models", exist_ok=True)

# Export the pickle file
joblib.dump(lr_clf, output_path)

print(f"\n[SUCCESS] Model successfully saved to: {output_path}")