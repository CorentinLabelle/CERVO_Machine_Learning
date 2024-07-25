import numpy as np
from sklearn.decomposition import PCA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.base import BaseEstimator, TransformerMixin


class TimeAverage(BaseEstimator, TransformerMixin):
    """
    Perform the mean over time for each signal.
    """
    def __init__(self, start_sample=0, end_sample=None):
        self.start_sample = start_sample
        self.end_sample = end_sample

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        if X.ndim == 1:
            # Assume it is one sample at one time stamp
            # X: (nb_channels) --> (nb_channels, 1)
            X = np.expand_dims(X, axis=1)

        if X.ndim == 2:
            # Assume it is only one sample, add the first dimension
            # X: (nb_channels, nb_samples) --> (1, nb_channels, nb_samples)
            X = np.expand_dims(X, axis=0)

        if self.end_sample is None:
            self.end_sample = X.shape[2]

        return np.mean(X[:, :, self.start_sample:self.end_sample], axis=2)


class Squeeze(BaseEstimator, TransformerMixin):
    """
    Squeeze Signal
    """
    def __init__(self, start_sample=0, end_sample=None):
        self.start_sample = start_sample
        self.end_sample = end_sample

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.squeeze(X)


class TimeMaxPooling(BaseEstimator, TransformerMixin):
    """

    """
    def __init__(self, start_sample=0, end_sample=None):
        self.start_sample = start_sample
        self.end_sample = end_sample

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        if X.ndim == 2:
            # Assume it is only one sample, add the first dimension
            # X: (nb_channels, nb_samples) --> (1, nb_channels, nb_samples)
            X = np.expand_dims(X, axis=0)

        if self.end_sample is None:
            self.end_sample = X.shape[2]

        return np.max(X[:, :, self.start_sample:self.end_sample], axis=2)


class CustomPCA(BaseEstimator, TransformerMixin):
    """

    """
    def __init__(self, start_sample=0, end_sample=None):
        self.start_sample = start_sample
        self.end_sample = end_sample

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        if X.ndim == 2:
            # Assume it is only one sample, add the first dimension
            # X: (nb_channels, nb_samples) --> (1, nb_channels, nb_samples)
            X = np.expand_dims(X, axis=0)

        if self.end_sample is None:
            self.end_sample = X.shape[2]

        X = X[:, :, self.start_sample:self.end_sample]

        # Step 1: Reshape the data to 2D
        e, c, t = X.shape
        data_2d = X.reshape(e * c, t)

        # Step 2: Apply PCA
        pca = PCA(n_components=1)
        pca.fit(data_2d)

        # Transform the data
        data_pca = pca.transform(data_2d)

        # Step 3: Reshape back to original shape
        data_pca_3d = data_pca.reshape(e, c, 1)

        return data_pca_3d.squeeze()


class svd(BaseEstimator, TransformerMixin):
    """

    """
    def __init__(self, start_sample=0, end_sample=None):
        self.start_sample = start_sample
        self.end_sample = end_sample

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        if X.ndim == 2:
            # Assume it is only one sample, add the first dimension
            # X: (nb_channels, nb_samples) --> (1, nb_channels, nb_samples)
            X = np.expand_dims(X, axis=0)

        if self.end_sample is None:
            self.end_sample = X.shape[2]

        X = X[:, :, self.start_sample:self.end_sample]

        # Step 1: Reshape the data to 2D
        e, c, t = X.shape
        data_2d = X.reshape(e * c, t)

        u, sigma, vt = np.linalg.svd(data_2d, full_matrices=False)

        component = vt[0]

        data_pca_3d = component.reshape(e, c, 1)
        return data_pca_3d.squeeze()
