from fisher_vector import FisherVector
import numpy as np
fs = FisherVector()

X = []

for i in range(1000):
    X.append(np.random.randint(low=0, high=1000, size=(500, 128)))

fs.fit(X)
print(fs.databases.shape)
print(fs.fit_transform([X[0]]).shape)
