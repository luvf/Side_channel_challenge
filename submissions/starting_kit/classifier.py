from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

from sklearn.naive_bayes import GaussianNB


class Classifier(BaseEstimator):
    def __init__(self):
        self.clf = GaussianNB()

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        y_pred_proba = self.clf.predict_proba(X)
        return y_pred_proba
