import numpy as np
from sklearn.mixture import GaussianMixture as GMM
from sklearn.preprocessing import StandardScaler
from random import sample
import os
import pickle


class FisherVector:
    """
        Parameters
        ----------    
        gmm : instance of sklearn mixture.GMM object
            Gauassian mixture model of the descriptors.
        n_components : int 
    """

    def __init__(self, n_components=64):
        self.gmm = None
        self.n_components = n_components
        self.databases = None
        self.scaler = None

    def _build(self, X):
        self.gmm = GMM(n_components=self.n_components, covariance_type='diag')
        X_mat = np.vstack(X)
        if len(X_mat) < int(2e5):
            self.gmm.fit(X_mat)
        else:
            idx = sample(range(len(X_mat)), int(2e5))
            self.gmm.fit(X_mat[idx])

        self.databases = np.vstack([self.fisher_vector(data) for data in X])
        self.scaler = StandardScaler().fit(self.databases)
        self.databases = self.scaler.transform(self.databases)

    def fisher_vector(self, X):
        """
        Args:
            X : List[], shape = (N, D) 
            D : int,
                Số chiều của Local Descriptors như SIFT = 128
            N : int,
                Số lượng Local Descriptors
            Q : np array, shape (N, k)
                ----
            k : int,
                n_components của GMM    

        Returns:
            Paper:
                Modeling Spatial Layout with Fisher Vectors for Image Categorization.  
                weight + mean + variance : np array, shape (K + 2 * D * K, ) 

                Aggregating local descriptors into a compact image representation 
                mean : np array, shape = (D * k, )
        """
        X = np.vstack(X)
        N = X.shape[0]

        Q = self.gmm.predict_proba(X)

        # Compute the sufficient statistics of descriptors.
        Q_sum = np.sum(Q, 0)[:, np.newaxis] / N
        Q_xx = np.dot(Q.T, X) / N
        Q_xx_2 = np.dot(Q.T, X ** 2) / N

        # Compute derivatives with respect to mixing weights, means and variances.
        d_pi = Q_sum.squeeze() - self.gmm.weights_
        d_mu = Q_xx - Q_sum * self.gmm.means_
        d_sigma = (
            - Q_xx_2
            - Q_sum * self.gmm.means_ ** 2
            + Q_sum * self.gmm.covariances_
            + 2 * Q_xx * self.gmm.means_)

        # return np.hstack((d_pi, d_mu.flatten(), d_sigma.flatten()))
        return d_mu.flatten()

    def fit(self, X):
        self._build(X)
        return self

    def transform(self, X):
        o = []
        for data in X:
            fs_o = self.fisher_vector(data)
            fs_n = self.scaler.transform(fs_o.reshape(1, -1))
            o.append(fs_n)
        return np.vstack(o)

    def fit_transform(self, X):
        self._build(X)
        return self.transform(X)

    def save_model(self, dataset):
        if not os.path.exists("checkpoint"):
            os.makedirs("checkpoint")

        fish_checkpoint_path = os.path.join(
            "checkpoint", "fish_k{}_ds{}".format(self.n_components, dataset))

        if not os.path.exists(fish_checkpoint_path):
            os.makedirs(fish_checkpoint_path)

        np.savez_compressed(fish_checkpoint_path + "/data.npz",
                            databases=self.databases)

        with open(fish_checkpoint_path + "/gmm.pkl", "wb") as f:
            pickle.dump(self.gmm, f)
        with open(fish_checkpoint_path + "/scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)

    def load_model(self, path):
        data = np.load(path + "data.npz", allow_pickle=True)
        with open(path + "gmm.pkl", "rb") as f:
            gmm = pickle.load(f)
        with open(path + "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        self.databases = data['databases']
        self.gmm = gmm
        self.scaler = scaler
