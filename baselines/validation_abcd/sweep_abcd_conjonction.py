"""Validation empirique de la regle generalisee a quatre parametres libres
(a,b,c,d), cf. preuve_generale_abcd.tex -- l'equivalent, pour notre theorie,
de l'experience 2 de PCL (leur balayage sur p, Figure 3b) : verifier que la
frontiere theorique de convergence predit exactement l'endroit ou la
recuperation exacte de la conjonction s'effondre empiriquement.

On se place dans le cas le plus simple deja couvert par le Corollaire du
document (conjonction pure, entrees uniformes, a=c fixe) : dans ce cas la
theorie predit une frontiere nette et simple, valable pour tout m :

    recuperation garantie  <=>  d < b   (a=c=S fixe par ailleurs)

On fixe plusieurs couples (S,b) deja retenus dans l'etude hyperparametres
precedente, on fait varier d de 0 jusqu'au-dela de b, et on verifie que le
taux de recuperation exacte s'effondre pile au passage d=b -- pas de
boosting, pas de conditions, une seule clause, exactement le protocole PCL.

Usage: python3 sweep_abcd_conjonction.py
"""
import itertools
import time

import numpy as np
from numba import njit

n = 6
J_plus = [0, 1]
J_minus = [2]
m = len(J_plus) + len(J_minus)
N = 100


@njit(cache=True)
def entrainer_clause_abcd(X, y, N, a, b, c, d):
    """Regle a 4 parametres libres (Definition 1 de preuve_generale_abcd.tex).
    state[:,1]=v (litteral x_f), state[:,0]=vbar (litteral not x_f)."""
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


def formule(state, N, n):
    return sorted(
        ([j + 1 for j in range(n) if state[j, 1] > N])
        + ([-j - 1 for j in range(n) if state[j, 0] > N])
    )


def theorie_ok(a, b, c, d, m):
    """Conditions du Theoreme d'identification generalise (preuve_generale_abcd.tex),
    cas conjonction pure + entrees uniformes (Proposition prop:uniforme)."""
    cond_pertinent = a + (c + d) * 2 ** (1 - m) > c          # J+ et J- (meme formule)
    cond_irrelevant = a - c < (a + b - c - d) * 2 ** (-m)     # k non pertinent
    return cond_pertinent and cond_irrelevant


def main():
    X = np.array(list(itertools.product([0, 1], repeat=n)), dtype=np.int64)
    y = np.array([
        1 if all(row[j] == 1 for j in J_plus) and all(row[j] == 0 for j in J_minus) else 0
        for row in X
    ], dtype=np.int64)
    target = sorted([j + 1 for j in J_plus] + [-j - 1 for j in J_minus])
    print(f"Cible : {target}  (n={n}, m={m})")

    epochs = 200
    Xe = np.tile(X, (epochs, 1))
    ye = np.tile(y, epochs)
    R = 30

    # warm-up JIT
    entrainer_clause_abcd(Xe, ye, N, 1.0, 0.5, 1.0, 0.0)

    configs = [
        ("S=1.0, b=0.5", 1.0, 0.5),
        ("S=0.7, b=0.3", 0.7, 0.3),
        ("S=1.0, b=0.9", 1.0, 0.9),
    ]
    d_fracs = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0]

    t0 = time.time()
    for label, S, b in configs:
        a = c = S
        print(f"\n=== {label} (seuil theorique attendu : d < b = {b:.3f}) ===")
        print(f"{'d':>8} {'d/b':>6} {'theorie':>9} {'succes':>10}")
        for frac in d_fracs:
            d = frac * b
            pred = theorie_ok(a, b, c, d, m)
            succ = 0
            for _ in range(R):
                state = entrainer_clause_abcd(Xe, ye, N, a, b, c, d)
                if formule(state, N, n) == target:
                    succ += 1
            print(f"{d:>8.4f} {frac:>6.2f} {'OK' if pred else 'echec':>9} {succ:>6}/{R:<3}")
    print(f"\ntemps total : {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
