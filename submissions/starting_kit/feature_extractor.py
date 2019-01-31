from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
import numpy as np

class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.fitted = False
        self.mean_vectors = None
        # Indices for which we observed high variance in the data
        self.peaks_indices = [145, 667, 302, 406]


    def fit(self, X, y):
        self.y = y
        return self

    def transform(self, X):
        new_X = X.copy()
        new_X = np.concatenate([new_X.values[:,i-5:i+5] for i in self.peaks_indices], axis=1)
        # print(new_X.shape)
        if not self.fitted:
            self.mean_vectors = get_mean_vectors(new_X, self.y)
            self.fitted = True
        #pca_X = self.pca.transform(new_X)
        #new_X = pca_X
        dist_X = np.abs(pairwise_distances(new_X, self.mean_vectors, metric='euclidean'))
        new_X = np.concatenate((new_X, dist_X), axis=1)
        return new_X

def get_mean_vectors(X, y):
    mean_vectors = []

    for label in range(256):
        mean_vector = X[y==label].mean(axis=0)
        mean_vectors.append(mean_vector)

    return np.array(mean_vectors)

def get_sum(v, metric='euclidean'):
    return pairwise_distances(v, metric=metric).sum(axis=0)
