import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import accuracy_score
import joblib
from collections import Counter

# ==========================================
# 1. Model Definition
# ==========================================
class RuleBasedModel:
    def __init__(self, pred_thresh=3, x6_t=0.5, x1_t=1.0, x18_t=0.7, x6_w=2, x1_w=1, x18_w=1):
        # Final prediction threshold
        self.pred_thresh = pred_thresh
        
        # Internal rule thresholds
        self.x6_t = x6_t
        self.x1_t = x1_t
        self.x18_t = x18_t
        
        # Internal score weights
        self.x6_w = x6_w
        self.x1_w = x1_w
        self.x18_w = x18_w

    def predict_score(self, X):
        return X.apply(self._score_row, axis=1)

    def predict(self, X):
        scores = self.predict_score(X)
        return (scores >= self.pred_thresh).astype(int)

    def _score_row(self, row):
        score = 0
        if row["X6"] > self.x6_t:
            score += self.x6_w
        if row["X1"] < self.x1_t:
            score += self.x1_w
        if row["X18"] < self.x18_t:
            score += self.x18_w
        return score
        
    def pickle(self, path="../models/rule_based_model.pkl"):
        joblib.dump(self, path)


# ==========================================
# 2. Nested CV Implementation
# ==========================================
def nested_cv_rule_based(X, y, param_grid, outer_splits=5, inner_splits=3, metric=accuracy_score):
    """
    Implements nested cross-validation for the RuleBasedModel.
    Outer loop: Evaluates the model's generalized performance.
    Inner loop: Tunes the entire parameter grid (thresholds and weights).
    """
    outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=42)

    outer_scores = []
    best_params_list = []

    # Convert the dictionary grid into an iterable list of all parameter combinations
    grid = list(ParameterGrid(param_grid))
    print(f"Total hyperparameter combinations to search per fold: {len(grid)}")

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), 1):
        X_train_outer, X_test_outer = X.iloc[train_idx], X.iloc[test_idx]
        y_train_outer, y_test_outer = y.iloc[train_idx], y.iloc[test_idx]

        best_params = None
        best_inner_score = -1.0

        # Inner Loop: Find the best parameter combination for this fold
        for params in grid:
            inner_scores = []
            
            for inner_train_idx, inner_val_idx in inner_cv.split(X_train_outer, y_train_outer):
                X_val_inner = X_train_outer.iloc[inner_val_idx]
                y_val_inner = y_train_outer.iloc[inner_val_idx]

                # Initialize model with the current dictionary of parameters unpacked
                model = RuleBasedModel(**params)
                
                y_pred_inner = model.predict(X_val_inner)
                score = metric(y_val_inner, y_pred_inner)
                inner_scores.append(score)

            avg_inner_score = np.mean(inner_scores)
            
            if avg_inner_score > best_inner_score:
                best_inner_score = avg_inner_score
                best_params = params

        # Outer Loop: Evaluate the best parameters on the outer test set
        best_model_for_fold = RuleBasedModel(**best_params)
        y_pred_outer = best_model_for_fold.predict(X_test_outer)
        outer_score = metric(y_test_outer, y_pred_outer)
        
        outer_scores.append(outer_score)
        best_params_list.append(best_params)

        print(f"Fold {fold} | Best Params: {best_params} | Outer Test Score: {outer_score:.4f}")

    print("-" * 50)
    print(f"Overall Nested CV Score: {np.mean(outer_scores):.4f} ± {np.std(outer_scores):.4f}")
    
    return outer_scores, best_params_list


# ==========================================
# 3. Main Execution Block
# ==========================================
if __name__ == "__main__":
    print("Loading data...")
    try:
        X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = joblib.load("data/processed/3features.pkl")
    except FileNotFoundError:
        print("Data file not found. Please verify the path.")
        exit()
        
    y_train = y_train.astype(int)
    y_val = y_val.astype(int)
    y_test = y_test.astype(int)
    
    print("Combining Train and Val sets for Cross-Validation...")
    X_cv = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_cv = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)

    # ---------------------------------------------------------
    # Define the Search Space (The Parameter Grid)
    # Be mindful of the combinatorial explosion. 
    # Testing too many values will slow down the nested CV.
    # ---------------------------------------------------------
    param_grid = {
        "pred_thresh": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "x6_t": [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1 ,0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "x1_t": [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1 ,0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "x18_t": [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1 ,0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "x6_w": [ -5, -4, -3, -2, -1, 1, 2, 3, 4, 5],
        "x1_w": [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5],
        "x18_w": [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]
    }

    print("\nStarting Nested CV...")
    cv_scores, best_params_list = nested_cv_rule_based(
        X=X_cv, 
        y=y_cv, 
        param_grid=param_grid, 
        outer_splits=5, 
        inner_splits=3,
        metric=accuracy_score 
    )

    # Find the most frequently selected dictionary of parameters across folds
    # We convert dicts to sorted tuples so they can be counted by collections.Counter
    param_tuples = [tuple(sorted(p.items())) for p in best_params_list]
    most_common_tuple = Counter(param_tuples).most_common(1)[0][0]
    final_best_params = dict(most_common_tuple)
    
    print(f"\nMost frequently selected parameters across folds:\n{final_best_params}")

    print(f"\nApplying final parameters to unseen holdout X_test...")
    final_model = RuleBasedModel(**final_best_params)

    # Predict and score on the final holdout set
    y_test_pred = final_model.predict(X_test)
    test_score = accuracy_score(y_test, y_test_pred)

    print(f"Final Held-out Test Score: {test_score:.4f}")

    # Save the final optimized model
    model_save_path = "models/rule_based.pkl"
    final_model.pickle(model_save_path)
    print(f"Optimized model saved to: {model_save_path}")