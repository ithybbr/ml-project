from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
import joblib
class RuleBasedModel:
    def __init__(self, threshold=3):
        self.threshold = threshold

    def predict_score(self, X):
        return X.apply(self._score_row, axis=1)

    def predict(self, X):
        scores = self.predict_score(X)
        return (scores >= self.threshold).astype(int)

    def _score_row(self, row):
        score = 0
        if row["X6"] > 0.5:
            score += 2
        if row["X1"] < 1.0:
            score += 1
        if row["X18"] < 0.7:
            score += 1
        return score
    def pickle(self, path = "../models/rule_based_model.pkl"):
        joblib.dump(self, path)