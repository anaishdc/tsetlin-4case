"""Notre regle generalisee (a,b,c,d) + conditions aleatoires, SANS boosting
(alpha=1 uniforme, vote non pondere) sur Iris binaire -- meme protocole que
ablation_iris.py (M=300, K=2, total=2000, N=100), pour voir si le nouveau
degre de liberte d change quelque chose par rapport au resultat de
l'ancienne regle (67.07% +/- 6.28%, conditions seules sans boosting).

Usage: python3 notre_regle_abcd_conditions_iris.py
"""
import time

import numpy as np
from numba import njit

DATA_PATH = "../dpcl-classifier/dpcl_classifier/binary_iris.txt"


@njit(cache=True)
def entrainer_clause_abcd(X, y, active, N, a, b, c, d):
    n_ex, n_feat = X.shape
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
            if xf == 0 and yy == 0:
                if np.random.random() < a and v < 2 * N:
                    v += 1
                if np.random.random() < c and vbar > 1:
                    vbar -= 1
            elif xf == 0 and yy == 1:
                if np.random.random() < b and v > 1:
                    v -= 1
                if np.random.random() < d and vbar < 2 * N:
                    vbar += 1
            elif xf == 1 and yy == 0:
                if np.random.random() < c and v > 1:
                    v -= 1
                if np.random.random() < a and vbar < 2 * N:
                    vbar += 1
            else:
                if np.random.random() < d and v < 2 * N:
                    v += 1
                if np.random.random() < b and vbar > 1:
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


def entrainer_ensemble(X_train, Y_train, n_feat, M, K, total, a, b, c, d, N, rng):
    n_ex = len(Y_train)
    clauses = []

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
            clauses.append((None, active))
            continue

        Sm_pos = S_m[Y_train[S_m] == 1]
        Sm_neg = S_m[Y_train[S_m] == 0]

        if len(Sm_pos) > 0 and len(Sm_neg) > 0:
            n_pos = rng.binomial(total, 0.5)
            n_neg = total - n_pos
            dpos = rng.choice(Sm_pos, size=n_pos) if n_pos > 0 else np.empty(0, dtype=np.int64)
            dneg = rng.choice(Sm_neg, size=n_neg) if n_neg > 0 else np.empty(0, dtype=np.int64)
            draws = np.concatenate([dpos, dneg])
            rng.shuffle(draws)
        else:
            draws = rng.choice(S_m, size=total)

        Xd = X_train[draws]
        yd = Y_train[draws].astype(np.int64)
        state = entrainer_clause_abcd(Xd, yd, active, N, a, b, c, d)

        if not clause_nonempty(state, N, active):
            clauses.append((None, active))
            continue

        clauses.append((state, active))

    return clauses


def predict(clauses, X, N):
    scores = np.zeros(len(X))
    for state, active in clauses:
        if state is None:
            continue
        for i in range(len(X)):
            o = clause_output(state, N, X[i], active)
            scores[i] += (2 * o - 1)  # alpha=1 uniforme, vote non pondere
    return (scores > 0).astype(int)


def main():
    data = np.loadtxt(DATA_PATH).astype(np.int64)
    total, N = 2000, 100
    a, b, c, d = 1.0, 0.3, 1.0, 0.0
    runs = 50

    # K plus grand -> moins de litteraux libres par clause -> clauses plus
    # courtes (comme chez PCL) ; M augmente avec K pour compenser le fait
    # que chaque clause couvre moins de l'espace.
    variantes = {
        "K=2,  M=300":  (2, 300),
        "K=6,  M=300":  (6, 300),
        "K=8,  M=500":  (8, 500),
        "K=10, M=800":  (10, 800),
        "K=12, M=1500": (12, 1500),
        "K=14, M=3000": (14, 3000),
    }

    rng0 = np.random.default_rng(0)
    entrainer_ensemble(data[:20, :16], (data[:20, 16] == 1).astype(np.int64),
                        16, 2, 2, 50, a, b, c, d, N, rng0)

    print(f"Iris binaire, conditions seules SANS boosting, a={a},b={b},c={c},d={d}, total={total}, N={N}, {runs} runs\n")
    for nom, (K, M) in variantes.items():
        accs = np.empty(runs)
        t0 = time.time()
        for r in range(runs):
            rng = np.random.default_rng(4000 + r)
            perm = rng.permutation(len(data))
            dd = data[perm]
            n_train = int(0.7 * len(dd))
            X_train = dd[:n_train, :16]
            Y_train = (dd[:n_train, 16] == 1).astype(np.int64)
            X_test = dd[n_train:, :16]
            Y_test = (dd[n_train:, 16] == 1).astype(np.int64)

            clauses = entrainer_ensemble(X_train, Y_train, 16, M, K, total, a, b, c, d, N, rng)
            pred = predict(clauses, X_test, N)
            accs[r] = (pred == Y_test).mean()
        dt = time.time() - t0
        print(f"{nom:45s} : {100*accs.mean():.2f}% +/- {100*accs.std():.2f}%   ({dt:.1f}s / {runs} runs)")


if __name__ == "__main__":
    main()
