import numpy as np
from adc import ADC
from sklearn.cluster import KMeans
X = np.random.randint(size=(1000, 80), low=0, high=10)
adc = ADC()
# print(X)
adc._train(X)
print(adc.transform(X).shape)

X_test = np.random.randint(size=(1, 80), low=0, high=10)

indices = adc.transform(X_test)
# codebooks = shape [k, m, len]
codebooks = adc.centers
print(indices.shape, codebooks.shape)

reconstructed_vector = np.hstack(
    [codebooks[idx, i] for i, idx in enumerate(indices)])

print(reconstructed_vector.shape)
print(np.sum(reconstructed_vector - reconstructed_vector.reshape(1, 80).reshape(16, 5)))

print(X_test - reconstructed_vector.reshape(1, 80))
print(adc.transform(
    X_test - reconstructed_vector.reshape((1, -1))
))

X = np.random.randint(size=(1000, 80), low=0, high=10)

model = KMeans(n_clusters=8).fit(X)
center = model.cluster_centers_

predict = model.predict(X)
q_y = center[predict]
print(center.shape, predict.shape, q_y.shape)
