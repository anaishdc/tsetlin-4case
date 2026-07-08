#!/usr/bin/env python3
# -----------------------------------------------------------------
# Generateur du probleme du multiplexeur booleen (benchmark classique
# de la litterature Tsetlin Machine -- Granmo 2018 et suivants, et de
# la litterature XCS/genetique avant elle). k bits d'adresse + 2^k
# bits de donnees, n = k + 2^k. Label = valeur du bit de donnee
# selectionne par l'adresse (encodage binaire). Aucune feature seule
# n'est informative -- seule la combinaison adresse+donnee compte,
# meme signature "interaction combinatoire pure" que le XOR bruite.
# -----------------------------------------------------------------
import numpy as np

def gen_multiplexer(k_addr, n_samples, seed, noise=0.0):
    rng = np.random.RandomState(seed)
    n_data = 2 ** k_addr
    n = k_addr + n_data
    X = rng.randint(0, 2, size=(n_samples, n))
    addr = X[:, :k_addr]
    data = X[:, k_addr:]
    idx = np.zeros(n_samples, dtype=int)
    for b in range(k_addr):
        idx = idx * 2 + addr[:, b]
    y = data[np.arange(n_samples), idx].copy()
    if noise > 0:
        flip = rng.rand(n_samples) < noise
        y[flip] = 1 - y[flip]
    return X, y

def write(path, X, y):
    with open(path, "w") as f:
        for i in range(X.shape[0]):
            f.write(" ".join(str(v) for v in X[i]) + " " + str(y[i]) + "\n")

# 6-bit MUX : 2 bits d'adresse + 4 bits de donnees, n=6
Xtr, ytr = gen_multiplexer(2, 5000, seed=40000, noise=0.0)
Xte, yte = gen_multiplexer(2, 5000, seed=41000, noise=0.0)
write("Mux6TrainingData.txt", Xtr, ytr)
write("Mux6TestData.txt", Xte, yte)
print("Mux6 : n=6 (2 addr + 4 data), train=5000 test=5000, sans bruit")

# 11-bit MUX : 3 bits d'adresse + 8 bits de donnees, n=11
Xtr, ytr = gen_multiplexer(3, 5000, seed=40001, noise=0.0)
Xte, yte = gen_multiplexer(3, 5000, seed=41001, noise=0.0)
write("Mux11TrainingData.txt", Xtr, ytr)
write("Mux11TestData.txt", Xte, yte)
print("Mux11 : n=11 (3 addr + 8 data), train=5000 test=5000, sans bruit")
