"""Balayage de S (et a) pour l'architecture TM classique (clauses +/-, pas
de conditionnement) sur XOR -- teste si reintroduire du hasard sur la
branche dominante (S<1) permet aux clauses de la meme polarite de
diverger, la ou S=1.0 les faisait toutes collapser vers la meme structure.

Usage: python3 sweep_S_archi_tm_xor.py
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


def load(path, n_feat):
    data = np.loadtxt(path, dtype=np.int64)
    return data[:, :n_feat], data[:, n_feat]


def main():
    n_feat = 12
    X_train, Y_train = load("../data/NoisyXORTrainingData.txt", n_feat)
    X_test, Y_test = load("../data/NoisyXORTestData.txt", n_feat)

    M, epochs, N = 20, 200, 50
    runs = 8

    configs = [
        (1.0, 0.35), (0.9, 0.35), (0.7, 0.35), (0.5, 0.35),
        (0.3, 0.35), (0.15, 0.35), (0.05, 0.35),
        (0.5, 0.1), (0.5, 0.5), (0.2, 0.2),
    ]

    entrainer_clause(X_train[:10], Y_train[:10].astype(np.int64), N, 1.0, 0.35)  # warm-up
    predict_all(np.full((1, n_feat, 2), N, dtype=np.int64), np.full((1, n_feat, 2), N, dtype=np.int64), X_test[:5], N)

    print(f"XOR, architecture TM (M={M}, pas de conditionnement, {epochs} epochs), {runs} runs par config\n")
    for S, a in configs:
        accs = np.empty(runs)
        t0 = time.time()
        for r in range(runs):
            rng = np.random.default_rng(8000 + r)
            pos_states, neg_states = train_ensemble(X_train, Y_train, n_feat, M, epochs, S, a, N, rng)
            pred = predict_all(pos_states, neg_states, X_test, N)
            accs[r] = (pred == Y_test).mean()
        dt = time.time() - t0
        print(f"S={S:.2f} a={a:.2f} : {100*accs.mean():.2f}% +/- {100*accs.std():.2f}%   ({dt:.1f}s/{runs} runs)")


if __name__ == "__main__":
    main()
