from pathlib import Path
import warnings
import pandas as pd
import joblib
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
# 2. TRAIN XGBOOST
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

    # Save model
    joblib.dump(xgb_model, SAVE_DIR / "xgb_engineered_features.pkl")


if __name__ == "__main__":
    main()