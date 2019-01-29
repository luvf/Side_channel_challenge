from sklearn.base import BaseEstimator
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.utils import to_categorical

import numpy as np

class Classifier(BaseEstimator):
    def __init__(self):
        self.epochs=200
        self.batch_size=100
        self.nodes = 200
        self.model = Sequential()
        self.model.add(Dense(self.nodes, input_dim=700, activation='relu'))
        self.model.add(Dense(256, activation='softmax'))
        self.optimizer = RMSprop(lr=0.00001)
        self.model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        self.clf = self.model
    def fit(self, X, y):
        self.clf.fit(X, y=to_categorical(y, num_classes=256), batch_size=self.batch_size, verbose = 1, epochs=self.epochs)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        y_pred_proba = self.clf.predict_proba(X)
        return y_pred_proba
