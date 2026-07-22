"""Notre regle a 4 cas, branchee sur l'architecture CLASSIQUE de la TM,
testee sur XOR bruite (avant Iris, plus rapide) :
- M clauses au total, moitie polarite positive, moitie polarite negative
  (comme une vraie TM), PAS de conditionnement aleatoire (K=0) : chaque
  clause voit les 12 features et TOUT le dataset d'entrainement.
- Entrainement par epochs sur l'ensemble complet, pas de tirage pondere ni
  de boosting -- seule la regle de mise a jour differe de PCL/TM.
- Clause positive entrainee sur (x,y) ; clause negative sur (x,1-y).
- Score(x) = #clauses positives qui matchent x - #clauses negatives qui
  matchent x ; prediction = 1 si score>0.

Usage: python3 notre_regle_archi_tm_xor.py
"""
import time

import numpy as np
from numba import njit


@njit(cache=True)
def entrainer_clause(X, y, N, S, a):
    n_ex, n_feat = X.shape
    state = np.empty((n_feat, 2), dtype=np.int64)
    for f in range(n_feat):
        state[f, 0] = N
        state[f, 1] = N + 1
    for e in range(n_ex):
        yy = y[e]
        for f in range(n_feat):
            xf = X[e, f]
            v = state[f, 1]
            vbar = state[f, 0]
            if xf == 0:
                if yy == 0:
                    if np.random.random() < S and v < 2 * N:
                        v += 1
                    if np.random.random() < S and vbar > 1:
                        vbar -= 1
                else:
                    if np.random.random() < a and v > 1:
                        v -= 1
            else:
                if yy == 0:
                    if np.random.random() < S and v > 1:
                        v -= 1
                    if np.random.random() < S and vbar < 2 * N:
                        vbar += 1
                else:
                    if np.random.random() < a and vbar > 1:
                        vbar -= 1
            state[f, 1] = v
            state[f, 0] = vbar
    return state


def clause_output(state, N, x):
    n_feat = len(x)
    for f in range(n_feat):
        if state[f, 1] > N and x[f] == 0:
            return 0
        if state[f, 0] > N and x[f] == 1:
            return 0
    return 1


def train_ensemble(X_train, Y_train, M, epochs, S, a, N, rng):
    n_pos = M // 2
    n_neg = M - n_pos
    n_ex = len(Y_train)

    pos_clauses = []
    for _ in range(n_pos):
        order = rng.permutation(n_ex)
        Xe = np.tile(X_train[order], (epochs, 1))
        ye = np.tile(Y_train[order], epochs).astype(np.int64)
        pos_clauses.append(entrainer_clause(Xe, ye, N, S, a))

    neg_clauses = []
    Y_flip = 1 - Y_train
    for _ in range(n_neg):
        order = rng.permutation(n_ex)
        Xe = np.tile(X_train[order], (epochs, 1))
        ye = np.tile(Y_flip[order], epochs).astype(np.int64)
        neg_clauses.append(entrainer_clause(Xe, ye, N, S, a))

    return pos_clauses, neg_clauses


def predict(pos_clauses, neg_clauses, X, N):
    scores = np.zeros(len(X))
    for state in pos_clauses:
        for i in range(len(X)):
            scores[i] += clause_output(state, N, X[i])
    for state in neg_clauses:
        for i in range(len(X)):
            scores[i] -= clause_output(state, N, X[i])
    return (scores > 0).astype(int)


def load(path, n_feat):
    data = np.loadtxt(path, dtype=np.int64)
    return data[:, :n_feat], data[:, n_feat]


def main():
    n_feat = 12
    X_train, Y_train = load("../data/NoisyXORTrainingData.txt", n_feat)
    X_test, Y_test = load("../data/NoisyXORTestData.txt", n_feat)

    M, epochs, S, a, N = 20, 200, 1.0, 0.35, 50  # M=20 comme la config TM du papier pour XOR
    runs = 20

    entrainer_clause(X_train[:10], Y_train[:10].astype(np.int64), N, S, a)  # warm-up JIT

    accs = np.empty(runs)
    t0 = time.time()
    for r in range(runs):
        rng = np.random.default_rng(7000 + r)
        pos_clauses, neg_clauses = train_ensemble(X_train, Y_train, M, epochs, S, a, N, rng)
        pred = predict(pos_clauses, neg_clauses, X_test, N)
        accs[r] = (pred == Y_test).mean()
        print(f"run {r}: acc={100*accs[r]:.2f}%", flush=True)

    dt = time.time() - t0
    print(f"\nNotre regle, architecture TM (M={M}, {M//2}+/{M-M//2}-, pas de "
          f"conditionnement, {epochs} epochs), {runs} runs")
    print(f"Precision : {100*accs.mean():.2f}% +/- {100*accs.std():.2f}%")
    print(f"temps total : {dt:.1f}s")


if __name__ == "__main__":
    main()