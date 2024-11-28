from sklearn.cluster import KMeans
import numpy as np
from sklearn.preprocessing import normalize
from random import sample
import os
import pickle


class BOF:
    """

    Args:
        N : int,
            Số lượng samples Database
        X : List[[[]]] shape (N, n_descriptors, n_dimensions)
        k : int
            Số lượng cluster hay số chiều đầu ra của vector biểu diễn
        kmeans : instance of sklearn.cluster.KMeans
            KMeans object
        centers : np array, shape = (k, D)      
        idf : np array, shape = (k, )
            log(N / (1 + n_j))
            n_j : int
                Số lượng samples thuộc cluster j
            Nghịch đảo tần suất đối với mỗi cluster
    """

    def __init__(self, k=1000):
        self.k = k
        self.databases = None
        self.kmeans = None
        self.centers = None
        self.idf = None

    def _build(self, X):
        centers = []
        X_mat = np.vstack(X)
        if len(X_mat) < int(2e5):
            kmeans = KMeans(n_clusters=self.k).fit(X_mat)
        else:
            idx = sample(range(len(X_mat)), int(2e5))
            kmeans = KMeans(n_clusters=self.k).fit(X_mat[idx])
        centers = kmeans.cluster_centers_

        self.kmeans = kmeans
        self.centers = centers
        self.databases = self._build_database(X)
        self.idf = self.compute_idf()
        self.databases *= self.idf
        self.databases = normalize(self.databases, norm='l2', axis=1)

    def _build_database(self, X):
        databases = np.zeros(shape=(len(X), self.k))
        for i, data in enumerate(X):
            predicts = np.array(self.kmeans.predict(data))
            for pos in predicts:
                databases[i][pos] += 1
        return databases

    def compute_idf(self):
        N = self.databases.shape[0]
        document_frequency = np.sum(self.databases > 0, axis=0)
        idf = np.log((N + 1) / (1 + document_frequency)) + 1
        return idf

    def fit(self, X):
        self._build(X)
        return self

    def transform(self, X):
        if type(X) is not list:
            X = [X]
        tokenize = self._build_database(X)
        tokenize_idf = tokenize * self.idf
        tokenize_norm = normalize(tokenize_idf, norm='l2', axis=1)
        return tokenize_norm

    def fit_transform(self, X):
        self._build(X)
        return self.databases

    def save_model(self, dataset):
        if not os.path.exists("checkpoint"):
            os.makedirs("checkpoint")

        bof_checkpoint_path = os.path.join(
            "checkpoint", "bof_k{}_ds{}".format(self.k, dataset))

        if not os.path.exists(bof_checkpoint_path):
            os.makedirs(bof_checkpoint_path)

        np.savez_compressed(bof_checkpoint_path + "/data.npz",
                            databases=self.databases, centers=self.centers, idf=self.idf)

        with open(bof_checkpoint_path + "/vocabs.pkl", "wb") as f:
            pickle.dump(self.kmeans, f)

    def load_model(self, path):
        data = np.load(path + "data.npz", allow_pickle=True)
        with open(path + "vocabs.pkl", "rb") as f:
            vocabs = pickle.load(f)

        self.databases = data["databases"]
        self.centers = data["centers"]
        self.idf = data["idf"]
        self.kmeans = vocabs
