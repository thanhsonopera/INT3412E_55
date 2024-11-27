from bof import BOF
import numpy as np
X = []
for _ in range(20):
    X.append(np.random.randint(low=0, high=100, size=(100, 50)))
bof = BOF(k=10)
bof._build(X)

print(bof.transform(X[0]))
# arr = np.array([[1, 2, 3]])
# print(arr / np.sqrt(np.sum(np.power(arr, 2))),
#       normalize(arr, norm='l2', axis=1))
