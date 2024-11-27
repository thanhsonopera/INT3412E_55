import numpy as np
from sklearn.cluster import KMeans
import os
import pickle


class ADC:
    '''
    Args:
    ----------
        kmeans: List[]
            Danh sách models KMeans
        X : List[], Shape(N, d)
            Input
        databases: np Array, Shape(N, m)
            Giá trị sau khi Quantized của X
        m : int,
            Số lượng Subvectors
        k : int, [256, 1024]
            Số lượng cụm tạo từ điển
        bs : int, log2(k)
            Số bit của mỗi Subvectors
        d : int,
            Số chiều Vector biểu diễn Local Descriptors
        d / m : int,
            Số chiều của mỗi Subvectors
    '''

    def __init__(self, m=16, k=256):
        self.kmeans = None
        self.databases = None
        self.m = m
        self.k = k
        self.bs = np.log2(k)

    def _train(self, X):
        self.kmeans = []
        n, d = X.shape
        self.centers = np.empty(shape=(self.k, self.m, int(d / self.m)))

        for i in range(self.m):
            self.kmeans.append(KMeans(n_clusters=self.k).
                               fit(X[:, int(i * (d / self.m)): int((i + 1) * (d / self.m))]))
            self.centers[:, i, :] = self.kmeans[i].cluster_centers_
        self.databases = self.transform(X)

    def fit(self, X):
        self._train(X)
        return self

    def transform(self, X):
        n, d = X.shape
        storage = []
        for i in range(self.m):
            if self.bs <= 8:
                storage.append(self.kmeans[i].
                               predict(X[:, int(i * (d / self.m)): int((i + 1) * (d / self.m))])
                               .astype(np.uint8).reshape(-1, 1))
            else:
                storage.append(self.kmeans[i].
                               predict(X[:, int(i * (d / self.m)): int((i + 1) * (d / self.m))])
                               .astype(np.uint16).reshape(-1, 1))

        return np.hstack(storage)

    def predict(self, X):
        return np.argmin(self.predict_proba(X))

    def predict_proba(self, X):
        """
        Args:
            X : np array, Shape = (d, )
            LuT : Shape = (k, m)
                Look up Table
            centers : np array, Shape = (k, m, d/m)
                Tập hợp k giá trị cluster cho từng subvector
            distances : np array, Shape = (k, m, d/m)
                Khoảng cách subvectorj đến center ij
            databases : np Array, Shape(N, m)
                Giá trị sau khi Quantized của X

        Returns:
            output : np array, Shape = (N, )
                Kết quả dự đoán
        """
        x = X.reshape(1, self.m, -1)
        distances = x - self.centers
        # LuT = np.sum(distances * distances, axis=2)
        LuT = np.einsum("ijk,ijk->ij", distances, distances)  # EQ : 30

        scores = LuT[self.databases, range(
            LuT.shape[1])].sum(axis=1)  # EQ : 31
        return scores

    def fit_transform(self, X):
        self._train(X)
        return self.databases

    def save_model(self, dataset):
        if not os.path.exists("checkpoint"):
            os.makedirs("checkpoint")
        adc_checkpoint_path = os.path.join(
            "checkpoint", "adc_k{}_m{}_ds{}".format(self.k, self.m, dataset))
        if not os.path.exists(adc_checkpoint_path):
            os.makedirs(adc_checkpoint_path)
        np.savez_compressed(adc_checkpoint_path + "/data.npz",
                            databases=self.databases, centers=self.centers)

        with open(adc_checkpoint_path + "/vocabs.pkl", "wb") as f:
            pickle.dump(self.kmeans, f)

    def load_model(self, path):
        data = np.load(path + "data.npz", allow_pickle=True)
        with open(path + "vocabs.pkl", "rb") as f:
            vocabs = pickle.load(f)
        self.databases = data["databases"]
        self.centers = data["centers"]
        self.kmeans = vocabs
