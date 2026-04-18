import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
import warnings
def predict_default_score(row):
    score = 0

    if row["X6"] > 0.5:
        score += 2
    if row["X1"] < 1.0:
        score += 1
    if row["X18"] < 0.7:
        score += 1

    return score
def always_predict_default(row):
    return 1
def always_predict_notdefault(row):
    return 0

def score_based_model(X_test):
    y_score = X_test.apply(predict_default_score, axis=1)
    y_pred = (y_score >= 3).astype(int)
    return y_pred
def always_default_model(X_test):
    y_always_default = X_test.apply(always_predict_default, axis=1)
    return y_always_default
def always_notdefault_model(X_test):
    y_always_notdefault = X_test.apply(always_predict_notdefault, axis=1)
    return y_always_notdefault

if __name__ == "__main__":
    X_test = pd.read_csv("../data/processed/X_test.csv")
    X_train = pd.read_csv("../data/processed/X_train.csv")
    X_val = pd.read_csv("../data/processed/X_val.csv")
    y_train = pd.read_csv("../data/processed/y_train.csv")
    y_val = pd.read_csv("../data/processed/y_val.csv")
    y_test = pd.read_csv("../data/processed/y_test.csv")
    warnings.filterwarnings("ignore", category=UserWarning)
    y_score = X_test.apply(predict_default_score, axis=1)
    y_pred = (y_score >= 3).astype(int)
    print(classification_report(y_test['Y'], y_pred))
    y_always_default = X_test.apply(always_predict_default, axis=1)
    print(classification_report(y_test['Y'], y_always_default))
    y_always_notdefault = X_test.apply(always_predict_notdefault, axis=1)
    print(classification_report(y_test['Y'], y_always_notdefault))