
# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import rampwf as rw
import h5py

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import precision_score, recall_score

#-----------------------------------------------------------------------
problem_title = 'Side Channel Attack Challenge'
_target_column_name = 'Churn'
_prediction_label_names = [0, 1]
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names)

# An object implementing the workflow
workflow = rw.workflows.FeatureExtractorClassifier()

#-----------------------------------------------------------------------
# Define custom score metrics for the churner class
class AUTR(rw.score_types.classifier_base.ClassifierBaseScoreType):

    def __init__(self, name='autr', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):pass
        #return precision_score(y_true, y_pred)

score_types = [AUTR]

#-----------------------------------------------------------------------
def get_cv(X, y):
    """Returns stratified randomized folds."""
    cv = StratifiedShuffleSplit(n_splits=10, test_size=0.25, random_state=57)
    return cv.split(X,y)

def _read_data(path, f_name):pass

def get_train_data(path='.'):pass

def get_test_data(path='.'):pass
