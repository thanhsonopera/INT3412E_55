import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from random import sample
from numpy.linalg import norm


class VLAD:

    def __init__(self, k=256, n_vocabs=1, norming="original", lcs=False, alpha=0.2, verbose=True):
        """Initialize VLAD-object

        Notes
        -----

        Hyperparameters have to be set, even if `centers` and `qs` are set externally.
        """
        self.k = k
        self.n_vocabs = n_vocabs
        self.norming = norming
        self.vocabs = None
        self.centers = None
        self.databases = None
        self.lcs = lcs
        self.alpha = alpha
        self.qs = None
        self.verbose = verbose

    def fit(self, X):
        """Fit Visual Vocabulary

        Parameters
        ----------
        X : list(array)
            List of image descriptors

        Returns
        -------
        self : VLAD
            Fitted object
        """
        X_mat = np.vstack(X)
        print(X_mat.shape[1])
        self.vocabs = []
        self.centers = []  # Is a list of `n_vocabs` np.arrays. Can be set externally without fitting
        self.qs = []  # Is a list of `n_vocabs` lists of length `k` of np.arrays. Can be set externally without fitting
        for i in range(self.n_vocabs):
            if self.verbose is True:
                print(f"Training vocab #{i+1}")
            if self.verbose is True:
                print(f"Training KMeans...")
            if len(X_mat) < int(2e5):
                self.vocabs.append(KMeans(n_clusters=self.k).fit(X_mat))
            else:
                idx = sample(range(len(X_mat)), int(2e5))
                self.vocabs.append(KMeans(n_clusters=self.k).fit(X_mat[idx]))
            self.centers.append(self.vocabs[i].cluster_centers_)
            if self.lcs is True and self.norming == "RN":
                if self.verbose is True:
                    print("Finding rotation-matrices...")
                predicted = self.vocabs[i].predict(X_mat)
                qsi = []
                for j in range(self.k):
                    q = PCA(n_components=X_mat.shape[1]).fit(
                        X_mat[predicted == j]).components_
                    qsi.append(q)
                self.qs.append(qsi)
        self.databases = self._extract_vlads(X)
        return self

    def transform(self, X):
        """Transform the input-tensor to a matrix of VLAD-descriptors

        Parameters
        ----------
        X : list(array)
            List of image-descriptors

        Returns
        -------
        vlads : array, shape (n, d * self.k)
            The transformed VLAD-descriptors
        """
        vlads = self._extract_vlads(X)
        return vlads

    def fit_transform(self, X):
        """Fit the model and transform the input-data subsequently

        Parameters
        ----------
        X : list(array)
            List of image-descriptors

        Returns
        -------
        vlads : array, shape (n, d * self.k)
            The transformed VLAD-descriptors
        """
        _ = self.fit(X)
        vlads = self.transform(X)
        return vlads

    def refit(self, X):
        """Refit the Visual Vocabulary

        Uses the already learned cluster-centers as in initial values for
        the KMeans-models

        Parameters
        ----------
        X : array
            The database used to refit the visual vocabulary

        Returns
        -------
        self : VLAD
            Refitted object
        """
        self.vocabs = []
        self.centers = []

        for i in range(self.n_vocabs):
            self.vocabs.append(KMeans(n_clusters=self.k, init=self.centers).fit(X.transpose((2, 0, 1))
                                                                                .reshape(-1, X.shape[1])))
            self.centers.append(self.vocabs[i].cluster_centers_)

        self.databases = self._extract_vlads(X)
        return self

    def predict(self, desc):
        """Predict class of given descriptor-matrix

        Parameters
        ----------
        desc : array
            A descriptor-matrix (m x d)

        Returns
        -------
        ``argmax(self.predict_proba(desc))`` : array
        """
        return np.argmax(self.predict_proba(desc))

    def predict_proba(self, desc):
        """Predict class of given descriptor-matrix, return probability

        Parameters
        ----------
        desc : array
            A descriptor-matrix (m x d)

        Returns
        -------
        ``self.databases @ vlad``
            The similarity for all database-classes
        """
        vlad = self._vlad(desc)  # Convert to VLAD-descriptor
        # Similarity between L2-normed vectors is defined as dot-product
        return self.databases @ vlad

    def _vlad(self, X):
        """Construct the actual VLAD-descriptor from a matrix of local descriptors

        Parameters
        ----------
        X : array
            Descriptor-matrix for a given image

        Returns
        -------
        ``V.flatten()`` : array
            The VLAD-descriptor
        """
        np.seterr(invalid='ignore',
                  divide='ignore')  # Division with 0 encountered below
        vlads = []

        for j in range(self.n_vocabs):  # Compute for multiple vocabs
            # predicted = self.vocabs[j].predict(X)  # Commented out in favor of line below (No dependency on actual vocab, but only on centroids)
            predicted = norm(
                X - self.centers[j][:, None, :], axis=-1).argmin(axis=0)
            _, d = X.shape
            V = np.zeros((self.k, d))  # Initialize VLAD-Matrix

            # Computing residuals
            if self.norming == "RN":
                curr = X - self.centers[j][predicted]
                curr /= norm(curr, axis=1)[:, None]
                # Untenstehendes kann noch vektorisiert werden

                for i in range(self.k):
                    V[i] = np.sum(curr[predicted == i], axis=0)
                    if self.lcs is True:
                        # Equivalent to multiplication in  summation above
                        V[i] = self.qs[j][i] @ V[i]
            else:
                for i in range(self.k):
                    V[i] = np.sum(X[predicted == i] -
                                  self.centers[j][i], axis=0)

            # Norming
            if self.norming in ("intra", "RN"):
                # L2-normalize every sum of residuals
                V /= norm(V, axis=1)[:, None]
                # Some of the rows contain 0s. np.nan will be inserted when dividing by 0!
                np.nan_to_num(V, copy=False)

            if self.norming in ("original", "RN"):
                V = self._power_law_norm(V)

            V /= norm(V)  # Last L2-norming
            V = V.flatten()
            vlads.append(V)
        vlads = np.concatenate(vlads)
        vlads /= norm(vlads)  # Not on axis, because already flat
        return vlads

    def _extract_vlads(self, X):
        """Extract VLAD-descriptors for a number of images

        Parameters
        ----------
        X : list(array)
            List of image-descriptors

        Returns
        -------
        database : array
            Database of all VLAD-descriptors for the given Tensor
        """
        vlads = []
        for x in X:
            vlads.append(self._vlad(x))

        database = np.vstack(vlads)
        return database

    def _add_to_database(self, vlad):
        """Add a given VLAD-descriptor to the database

        Parameters
        ----------
        vlad : array
            The VLAD-descriptor that should be added to the database

        Returns
        -------
        ``None``
        """
        self.databases = np.vstack((self.databases, vlad))

    def _power_law_norm(self, X):
        """Perform power-Normalization on a given array

        Parameters
        ----------
        X : array
            Array that should be normalized

        Returns
        -------
        normed : array
            Power-normalized array
        """
        normed = np.sign(X) * np.abs(X)**self.alpha
        return normed

    def __repr__(self):
        return f"VLAD(k={self.k}, norming=\"{self.norming}\")"

    def __str__(self):
        return f"VLAD(k={self.k}, norming=\"{self.norming}\")"
