from adc import CQ
import numpy as np
X = np.random.randint(size=(1000, 80), low=0, high=10)

model = CQ(k=8).fit(X)

print(model.residual(X).shape)
