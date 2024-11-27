import numpy as np
from sklearn.cluster import KMeans


class CQ:
    def __init__(self, k=16):
        self.kmeans = None
        self.databases = None
        self.centroids = None
        self.k = k

    def fit(self, X):
        self.kmeans = KMeans(n_clusters=self.k).fit(X)
        self.centroids = self.kmeans.cluster_centers_
        self.databases = self.transform(X)
        return self

    def transform(self, X):
        return self.kmeans.predict(X)

    def fit_transform(self, X):
        self.fit(X)
        return self.databases

    def residual(self, X):
        return X - self.centroids[self.transform(X)]


X = np.random.randint(size=(1000, 80), low=0, high=10)

model = CQ(k=8).fit(X)

print(model.residual(X).shape)
