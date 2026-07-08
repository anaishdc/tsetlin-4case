#!/usr/bin/env python3
# -----------------------------------------------------------------
# Generateur de parite bruitee a k bits (generalisation directe du
# Noisy XOR existant, qui est une parite a 2 bits). n=12 features
# au total (k bits informatifs + le reste en bruit pur), meme
# convention que NoisyXORTrainingData.txt : 40% de bruit sur le
# label du train, test propre.
# -----------------------------------------------------------------
import numpy as np

def gen_parity(k, n_total, n_samples, seed, train_noise=0.0):
    rng = np.random.RandomState(seed)
    X = rng.randint(0, 2, size=(n_samples, n_total))
    y = X[:, :k].sum(axis=1) % 2
    if train_noise > 0:
        flip = rng.rand(n_samples) < train_noise
        y = y.copy()
        y[flip] = 1 - y[flip]
    return X, y

def write(path, X, y):
    with open(path, "w") as f:
        for i in range(X.shape[0]):
            f.write(" ".join(str(v) for v in X[i]) + " " + str(y[i]) + "\n")

n_total = 12

for k, seed_base in [(3, 60000), (4, 61000)]:
    Xtr, ytr = gen_parity(k, n_total, 5000, seed=seed_base, train_noise=0.40)
    Xte, yte = gen_parity(k, n_total, 5000, seed=seed_base + 1, train_noise=0.0)
    write(f"Parity{k}TrainingData.txt", Xtr, ytr)
    write(f"Parity{k}TestData.txt", Xte, yte)
    print(f"Parity{k} : n={n_total} ({k} bits informatifs + {n_total-k} bruit), "
          f"train=5000 (40% bruit label) test=5000 (propre)")
