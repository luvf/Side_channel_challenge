from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator

import numpy as np

class Classifier(BaseEstimator):
    def __init__(self):
        self.clf = RandomForestClassifier(
            n_estimators=10, max_leaf_nodes=10, random_state=42)

    def fit(self, X, y):
    	print('aaa')
        #self.clf.fit(X, y)

    def predict_proba(self, X):
        out = np.zeros((len(X),256))+1/256
        return out
        #y_pred_proba = self.clf.predict_proba(X)
        #return y_pred_proba
