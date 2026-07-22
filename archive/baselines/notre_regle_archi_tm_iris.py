"""Notre regle a 4 cas, branchee sur l'architecture CLASSIQUE de la TM
(clauses +/- , pas de conditionnement, toutes les clauses voient tout le
dataset), testee sur Iris binaire (protocole PCL exact, meme donnees que
comparaison_iris_binaire.py) -- comparable au chiffre TM publie par PCL
(95.0%) et a notre propre reproduction de PCL (92.22%).

Usage: python3 notre_regle_archi_tm_iris.py
"""
import time

import numpy as np
from numba import njit

DATA_PATH = "dpcl-classifier/dpcl_classifier/binary_iris.txt"


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


@njit(cache=True)
def predict_all(pos_states, neg_states, X, N):
    n_pos = pos_states.shape[0]
    n_neg = neg_states.shape[0]
    n_feat = X.shape[1]
    n_ex = X.shape[0]
    preds = np.empty(n_ex, dtype=np.int64)
    for i in range(n_ex):
        score = 0
        for c in range(n_pos):
            fire = True
            for f in range(n_feat):
                if pos_states[c, f, 1] > N and X[i, f] == 0:
                    fire = False
                    break
                if pos_states[c, f, 0] > N and X[i, f] == 1:
                    fire = False
                    break
            if fire:
                score += 1
        for c in range(n_neg):
            fire = True
            for f in range(n_feat):
                if neg_states[c, f, 1] > N and X[i, f] == 0:
                    fire = False
                    break
                if neg_states[c, f, 0] > N and X[i, f] == 1:
                    fire = False
                    break
            if fire:
                score -= 1
        preds[i] = 1 if score > 0 else 0
    return preds


def train_ensemble(X_train, Y_train, n_feat, M, epochs, S, a, N, rng):
    n_pos = M // 2
    n_neg = M - n_pos
    n_ex = len(Y_train)

    pos_states = np.empty((n_pos, n_feat, 2), dtype=np.int64)
    for c in range(n_pos):
        order = rng.permutation(n_ex)
        Xe = np.tile(X_train[order], (epochs, 1))
        ye = np.tile(Y_train[order], epochs).astype(np.int64)
        pos_states[c] = entrainer_clause(Xe, ye, N, S, a)

    neg_states = np.empty((n_neg, n_feat, 2), dtype=np.int64)
    Y_flip = 1 - Y_train
    for c in range(n_neg):
        order = rng.permutation(n_ex)
        Xe = np.tile(X_train[order], (epochs, 1))
        ye = np.tile(Y_flip[order], epochs).astype(np.int64)
        neg_states[c] = entrainer_clause(Xe, ye, N, S, a)

    return pos_states, neg_states


def main():
    data = np.loadtxt(DATA_PATH).astype(np.int64)
    n_feat = 16
    M, epochs, S, a, N = 16, 100, 1.0, 0.3, 100  # M=16 comme PCL/leur TM
    runs = 50

    entrainer_clause(data[:10, :16], (data[:10, 16] == 1).astype(np.int64), N, S, a)
    predict_all(np.full((1, n_feat, 2), N, dtype=np.int64), np.full((1, n_feat, 2), N, dtype=np.int64), data[:5, :16], N)

    accs = np.empty(runs)
    t0 = time.time()
    for r in range(runs):
        rng = np.random.default_rng(9000 + r)
        perm = rng.permutation(len(data))
        d = data[perm]
        n_train = int(0.7 * len(d))
        X_train = d[:n_train, :16]
        Y_train = (d[:n_train, 16] == 1).astype(np.int64)
        X_test = d[n_train:, :16]
        Y_test = (d[n_train:, 16] == 1).astype(np.int64)

        pos_states, neg_states = train_ensemble(X_train, Y_train, n_feat, M, epochs, S, a, N, rng)
        pred = predict_all(pos_states, neg_states, X_test, N)
        accs[r] = (pred == Y_test).mean()

    dt = time.time() - t0
    print(f"Notre regle, architecture TM (M={M}, {M//2}+/{M-M//2}-, pas de "
          f"conditionnement, {epochs} epochs), Iris binaire, {runs} runs")
    print(f"Precision : {100*accs.mean():.2f}% +/- {100*accs.std():.2f}%")
    print(f"temps total : {dt:.1f}s")


if __name__ == "__main__":
    main()
