"""Comparaison PCL / notre METHODE COMPLETE (conditions aleatoires + AdaBoost
+ regle a 4 cas, exactement l'algorithme de la Section 5 du papier) sur le
protocole Iris binaire exact de PCL (PCL_BinaryIRIS.py, AAAI-25) : classe 1
(versicolor) vs reste, 16 features binaires, split 70/30 aleatoire, 100
epochs pour PCL (16 clauses, p_i~U(0.6,0.8)).

On ne teste plus "la regle seule sans boosting" (cf. discussion : sans
conditions ni boosting, la regle n'a aucun mecanisme de diversite entre
clauses et degenere -- toutes les clauses convergent vers la meme structure).
On compare donc PCL tel qu'il est reellement utilise a notre methode telle
qu'elle est reellement utilisee.

Usage: python3 comparaison_iris_binaire_complet.py
"""
import time

import numpy as np
from numba import njit

from pcl_iris_officiel import PCL as pcl_fit_officiel
from pcl_iris_officiel import Accuracy as accuracy_officiel

DATA_PATH = "dpcl-classifier/dpcl_classifier/binary_iris.txt"


@njit(cache=True)
def entrainer_clause(X, y, active, N, S, a):
    """X: (n_ex, n_features) 0/1, y: (n_ex,) 0/1, active: (n_features,) bool
    (True = automate libre, False = gele/exclu). Retourne l'etat (n,2)."""
    n = X.shape[0]
    n_feat = X.shape[1]
    state = np.full((n_feat, 2), N, dtype=np.int64)
    for f in range(n_feat):
        if active[f]:
            state[f, 0] = N
            state[f, 1] = N + 1
    for e in range(n):
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
        v_inc = state[f, 1] > N
        vbar_inc = state[f, 0] > N
        if v_inc and x[f] == 0:
            return 0
        if vbar_inc and x[f] == 1:
            return 0
    return 1


def clause_nonempty(state, N, active):
    for f in range(len(active)):
        if not active[f]:
            continue
        if state[f, 1] > N or state[f, 0] > N:
            return True
    return False


def entrainer_ensemble(X_train, Y_train, n_feat, M, K, total, S, a, N, rng):
    n_ex = len(Y_train)
    w = np.full(n_ex, 1.0 / n_ex)
    clauses = []  # (state, F_idx set as active mask, freeze values, alpha)
    idx_pos = np.where(Y_train == 1)[0]
    idx_neg = np.where(Y_train == 0)[0]

    for m in range(M):
        F = rng.choice(n_feat, size=K, replace=False)
        z = rng.integers(0, 2, size=K)
        active = np.ones(n_feat, dtype=np.bool_)
        active[F] = False

        mask = np.ones(n_ex, dtype=bool)
        for k in range(K):
            mask &= (X_train[:, F[k]] == z[k])
        S_m = np.where(mask)[0]
        if len(S_m) == 0:
            clauses.append((None, active, 0.0))
            continue

        Sm_pos = S_m[Y_train[S_m] == 1]
        Sm_neg = S_m[Y_train[S_m] == 0]

        if len(Sm_pos) > 0 and len(Sm_neg) > 0:
            n_pos_draws = rng.binomial(total, 0.5)
            n_neg_draws = total - n_pos_draws
            wpos = w[Sm_pos]; wpos = wpos / wpos.sum()
            wneg = w[Sm_neg]; wneg = wneg / wneg.sum()
            draws_pos = rng.choice(Sm_pos, size=n_pos_draws, p=wpos) if n_pos_draws > 0 else np.empty(0, dtype=np.int64)
            draws_neg = rng.choice(Sm_neg, size=n_neg_draws, p=wneg) if n_neg_draws > 0 else np.empty(0, dtype=np.int64)
            draws = np.concatenate([draws_pos, draws_neg])
            rng.shuffle(draws)
        else:
            wpool = w[S_m]; wpool = wpool / wpool.sum()
            draws = rng.choice(S_m, size=total, p=wpool)

        Xd = X_train[draws]
        yd = Y_train[draws].astype(np.int64)
        state = entrainer_clause(Xd, yd, active, N, S, a)

        if not clause_nonempty(state, N, active):
            clauses.append((None, active, 0.0))
            continue

        wrong = np.array([clause_output(state, N, X_train[i], active) != Y_train[i] for i in S_m])
        err = (w[S_m] * wrong).sum() / w[S_m].sum()
        err = min(max(err, 1e-6), 1 - 1e-6)
        alpha = 0.5 * np.log((1 - err) / err)
        for j, i in enumerate(S_m):
            w[i] *= np.exp(alpha) if wrong[j] else np.exp(-alpha)
        w /= w.sum()

        clauses.append((state, active, alpha, F, z))

    return clauses


def predict(clauses, X, N):
    scores = np.zeros(len(X))
    for entry in clauses:
        if entry[0] is None:
            continue
        state, active, alpha, F, z = entry
        mask = np.ones(len(X), dtype=bool)
        for k in range(len(F)):
            mask &= (X[:, F[k]] == z[k])
        for i in np.where(mask)[0]:
            o = clause_output(state, N, X[i], active)
            scores[i] += alpha * (2 * o - 1)
    return (scores > 0).astype(int)


def main():
    data = np.loadtxt(DATA_PATH).astype(np.int32)
    states_pcl, epochs_pcl, clauses_pcl = 10000, 100, 16
    pl, pu = 0.6, 0.8

    M, K, total, S, a, N = 300, 2, 2000, 1.0, 0.3, 100
    runs = 50

    accs_pcl = np.empty(runs)
    accs_notre = np.empty(runs)

    y0 = (data[:10, 16] == 1).astype(np.int32)
    probs0 = np.random.uniform(pl, pu, clauses_pcl)
    pcl_fit_officiel(16, 1, data[:10, :16], y0, clauses_pcl, states_pcl, probs0)
    entrainer_clause(data[:10, :16].astype(np.int64), y0.astype(np.int64),
                      np.ones(16, dtype=np.bool_), N, S, a)

    t0 = time.time()
    for r in range(runs):
        rng = np.random.default_rng(2000 + r)
        perm = rng.permutation(len(data))
        d = data[perm]
        n_train = int(0.7 * len(d))
        X_train = d[:n_train, :16]
        Y_train = (d[:n_train, 16] == 1).astype(np.int32)
        X_test = d[n_train:, :16]
        Y_test = (d[n_train:, 16] == 1).astype(np.int32)

        probs = np.random.uniform(pl, pu, clauses_pcl)
        state_pcl = pcl_fit_officiel(16, epochs_pcl, X_train, Y_train, clauses_pcl, states_pcl, probs)
        formulas = []
        for i in range(clauses_pcl):
            c = []
            for j in range(16):
                if state_pcl[i, j, 1] > states_pcl:
                    c.append(j + 1)
                if state_pcl[i, j, 0] > states_pcl:
                    c.append(-j - 1)
            formulas.append(c)
        accs_pcl[r] = accuracy_officiel(X_test, formulas, 16, Y_test)

        clauses = entrainer_ensemble(X_train, Y_train, 16, M, K, total, S, a, N, rng)
        pred = predict(clauses, X_test, N)
        accs_notre[r] = (pred == Y_test).mean()

        print(f"run {r}: PCL={100*accs_pcl[r]:.2f}%  notre={100*accs_notre[r]:.2f}%", flush=True)

    dt = time.time() - t0
    print(f"\nIris binaire, {runs} runs")
    print(f"PCL          : {100*accs_pcl.mean():.2f}% +/- {100*accs_pcl.std():.2f}%")
    print(f"Notre methode complete (M={M},K={K},total={total}) : "
          f"{100*accs_notre.mean():.2f}% +/- {100*accs_notre.std():.2f}%")
    print(f"temps total : {dt:.1f}s")


if __name__ == "__main__":
    main()
