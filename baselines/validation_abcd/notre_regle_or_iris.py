"""Notre regle a 4 parametres libres (a,b,c,d), combinee EXACTEMENT comme
PCL le fait sur Binary IRIS (section "PCL on Binary IRIS" du papier AAAI-25)
-- pas de boosting, pas de poids, pas de conditions aleatoires sur des
sous-ensembles de features : u clauses, chacune voit TOUT le jeu
d'entrainement (comme leur Algorithme 1), diversifiees uniquement par un
parametre tire aleatoirement par clause (b ~ U[0.3,1.0] ici -- plage qui
traverse le seuil de fragilite documente (alpha*~0.60 pour certaines
features d'Iris, cf etude_hyperparametres_S_alpha.tex), jouant le role de
leur p_i ~ U[0.6,0.8]), combinees par un simple OU logique :

    y_hat = C_1(x) OU C_2(x) OU ... OU C_u(x)

Protocole PCL exact : split 70/30 reshuffle a chaque run, 100 epochs,
nombre de clauses variable (0 a 15).

Usage: python3 notre_regle_or_iris.py
"""
import time

import numpy as np
from numba import njit

DATA_PATH = "../dpcl-classifier/dpcl_classifier/binary_iris.txt"


@njit(cache=True)
def entrainer_clause_abcd(X, y, N, a, b, c, d):
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


@njit(cache=True)
def clause_tire(state, x, N, n_feat):
    for f in range(n_feat):
        if state[f, 1] > N and x[f] == 0:
            return False
        if state[f, 0] > N and x[f] == 1:
            return False
    return True


@njit(cache=True)
def predict_all_or(states, X_test, N, n_feat, u):
    n_ex = X_test.shape[0]
    preds = np.zeros(n_ex, dtype=np.int64)
    for i in range(n_ex):
        fire = False
        for c in range(u):
            if clause_tire(states[c], X_test[i], N, n_feat):
                fire = True
                break
        preds[i] = 1 if fire else 0
    return preds


def train_or_ensemble(X_train, Y_train, n_feat, u, epochs, N, a, c, rng):
    Xe = np.tile(X_train, (epochs, 1))
    ye = np.tile(Y_train, epochs).astype(np.int64)
    states = np.empty((max(u, 1), n_feat, 2), dtype=np.int64)
    for i in range(u):
        b = rng.uniform(0.3, 1.0)
        d = 0.0
        states[i] = entrainer_clause_abcd(Xe, ye, N, a, b, c, d)
    return states


def main():
    data = np.loadtxt(DATA_PATH).astype(np.int64)
    n_feat = 16
    N = 100
    a = c = 1.0
    epochs = 100
    runs = 50
    u_values = [1, 2, 4, 6, 8, 10, 12, 15]

    # warm-up JIT
    entrainer_clause_abcd(data[:10, :16], (data[:10, 16] == 1).astype(np.int64), N, a, 0.3, c, 0.0)
    predict_all_or(np.full((1, n_feat, 2), N, dtype=np.int64), data[:5, :16], N, n_feat, 1)

    print("Notre regle (a,b,c,d) + combinaison OU (protocole PCL exact), Binary IRIS")
    print(f"{'u clauses':>10} {'accuracy':>18}")
    t0 = time.time()
    for u in u_values:
        accs = np.empty(runs)
        for r in range(runs):
            rng = np.random.default_rng(1000 + r)
            perm = rng.permutation(len(data))
            d_ = data[perm]
            n_train = int(round(0.7 * len(d_)))
            X_train = d_[:n_train, :16]
            Y_train = (d_[:n_train, 16] == 1).astype(np.int64)
            X_test = d_[n_train:, :16]
            Y_test = (d_[n_train:, 16] == 1).astype(np.int64)

            states = train_or_ensemble(X_train, Y_train, n_feat, u, epochs, N, a, c, rng)
            pred = predict_all_or(states, X_test, N, n_feat, u)
            accs[r] = (pred == Y_test).mean()
        print(f"{u:>10} {100*accs.mean():>8.2f}% +/- {100*accs.std():<6.2f}%")
    print(f"temps total : {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
