"""Ablation minimale sur XOR bruite : isoler la contribution des conditions
aleatoires et du boosting AdaBoost, a hyperparametres fixes (M,K,S,a,N,total
= configuration du papier, Section 6.1), en activant/desactivant chaque
mecanisme independamment.

4 variantes :
  A. regle seule       (K=0 : pas de conditions, alpha_m=1 : pas de boosting)
  B. +conditions seules (K=1, alpha_m=1 : pas de boosting)
  C. +boosting seul     (K=0 : pas de conditions, AdaBoost actif)
  D. modele complet     (K=1, AdaBoost actif -- config du papier)

Usage: python3 ablation_xor.py
"""
import time

import numpy as np
from numba import njit


@njit(cache=True)
def entrainer_clause(X, y, active, N, S, a):
    n_ex = X.shape[0]
    n_feat = X.shape[1]
    state = np.full((n_feat, 2), N, dtype=np.int64)
    for f in range(n_feat):
        if active[f]:
            state[f, 0] = N
            state[f, 1] = N + 1
    for e in range(n_ex):
        yy = y[e]
        for f in range(n_feat):
            if not active[f]:
                continue
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


def clause_output(state, N, x, active):
    for f in range(len(x)):
        if not active[f]:
            continue
        if state[f, 1] > N and x[f] == 0:
            return 0
        if state[f, 0] > N and x[f] == 1:
            return 0
    return 1


def clause_nonempty(state, N, active):
    for f in range(len(active)):
        if active[f] and (state[f, 1] > N or state[f, 0] > N):
            return True
    return False


def entrainer_ensemble(X_train, Y_train, n_feat, M, K, total, S, a, N, rng,
                        conditions=True, boosting=True):
    n_ex = len(Y_train)
    w = np.full(n_ex, 1.0 / n_ex)
    clauses = []

    for m in range(M):
        if conditions and K > 0:
            F = rng.choice(n_feat, size=K, replace=False)
            z = rng.integers(0, 2, size=K)
            active = np.ones(n_feat, dtype=np.bool_)
            active[F] = False
            mask = np.ones(n_ex, dtype=bool)
            for k in range(K):
                mask &= (X_train[:, F[k]] == z[k])
        else:
            F = np.empty(0, dtype=np.int64)
            z = np.empty(0, dtype=np.int64)
            active = np.ones(n_feat, dtype=np.bool_)
            mask = np.ones(n_ex, dtype=bool)

        S_m = np.where(mask)[0]
        if len(S_m) == 0:
            clauses.append((None, active, 0.0, F, z))
            continue

        Sm_pos = S_m[Y_train[S_m] == 1]
        Sm_neg = S_m[Y_train[S_m] == 0]

        if len(Sm_pos) > 0 and len(Sm_neg) > 0:
            n_pos = rng.binomial(total, 0.5)
            n_neg = total - n_pos
            wpos = w[Sm_pos]; wpos = wpos / wpos.sum()
            wneg = w[Sm_neg]; wneg = wneg / wneg.sum()
            dpos = rng.choice(Sm_pos, size=n_pos, p=wpos) if n_pos > 0 else np.empty(0, dtype=np.int64)
            dneg = rng.choice(Sm_neg, size=n_neg, p=wneg) if n_neg > 0 else np.empty(0, dtype=np.int64)
            draws = np.concatenate([dpos, dneg])
            rng.shuffle(draws)
        else:
            wpool = w[S_m]; wpool = wpool / wpool.sum()
            draws = rng.choice(S_m, size=total, p=wpool)

        Xd = X_train[draws]
        yd = Y_train[draws].astype(np.int64)
        state = entrainer_clause(Xd, yd, active, N, S, a)

        if not clause_nonempty(state, N, active):
            clauses.append((None, active, 0.0, F, z))
            continue

        if boosting:
            wrong = np.array([clause_output(state, N, X_train[i], active) != Y_train[i] for i in S_m])
            err = (w[S_m] * wrong).sum() / w[S_m].sum()
            err = min(max(err, 1e-6), 1 - 1e-6)
            alpha = 0.5 * np.log((1 - err) / err)
            for j, i in enumerate(S_m):
                w[i] *= np.exp(alpha) if wrong[j] else np.exp(-alpha)
            w /= w.sum()
        else:
            alpha = 1.0  # poids uniforme, pas de mise a jour de w

        clauses.append((state, active, alpha, F, z))

    return clauses


def predict(clauses, X, N, K):
    scores = np.zeros(len(X))
    for state, active, alpha, F, z in clauses:
        if state is None:
            continue
        if K > 0 and len(F) > 0:
            mask = np.ones(len(X), dtype=bool)
            for k in range(len(F)):
                mask &= (X[:, F[k]] == z[k])
        else:
            mask = np.ones(len(X), dtype=bool)
        for i in np.where(mask)[0]:
            o = clause_output(state, N, X[i], active)
            scores[i] += alpha * (2 * o - 1)
    return (scores > 0).astype(int)


def load(path, n_feat):
    data = np.loadtxt(path, dtype=np.int64)
    return data[:, :n_feat], data[:, n_feat]


def main():
    n_feat = 12
    X_train, Y_train = load("../data/NoisyXORTrainingData.txt", n_feat)
    X_test, Y_test = load("../data/NoisyXORTestData.txt", n_feat)

    M, K, total, S, a, N = 200, 1, 5000, 1.0, 0.35, 50
    runs = 10

    variantes = {
        "A. regle seule (sans conditions, sans boosting)": dict(conditions=False, boosting=False),
        "B. +conditions seules (sans boosting)": dict(conditions=True, boosting=False),
        "C. +boosting seul (sans conditions)": dict(conditions=False, boosting=True),
        "D. modele complet (conditions+boosting)": dict(conditions=True, boosting=True),
    }

    print(f"XOR bruite, M={M}, K={K}, total={total}, S={S}, a={a}, N={N}, {runs} runs\n")
    for nom, cfg in variantes.items():
        accs = np.empty(runs)
        t0 = time.time()
        for r in range(runs):
            rng = np.random.default_rng(5000 + r)
            clauses = entrainer_ensemble(X_train, Y_train, n_feat, M, K, total, S, a, N, rng, **cfg)
            pred = predict(clauses, X_test, N, K if cfg["conditions"] else 0)
            accs[r] = (pred == Y_test).mean()
        dt = time.time() - t0
        print(f"{nom:50s} : {100*accs.mean():.2f}% +/- {100*accs.std():.2f}%   ({dt:.1f}s / {runs} runs)")


if __name__ == "__main__":
    main()
