from vlad import VLAD
import numpy as np
X = []
for i in range(1000):
    X.append(np.random.randint(0, 255, (100, 128)))

vlad = VLAD(k=16, lcs=True, norming='RN')
vlad.fit(X)
vlad.save_model(1)

vlad2 = VLAD(k=16, lcs=True, norming='RN')
vlad2.load_model(
    "checkpoint/vlad_k{}_ds{}/".format(16, 1))

res1 = vlad.transform([X[0]])


res2 = vlad2.transform([X[0]])

print(np.sum(res1 - res2))
