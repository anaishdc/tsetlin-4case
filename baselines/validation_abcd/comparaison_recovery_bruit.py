"""Comparaison PCL / notre regle sur la recuperation d'une conjonction
SOUS BRUIT DE LABEL -- extension du protocole officiel de PCL (qui ne
teste que le cas sans bruit) pour voir comment les deux methodes se
degradent quand eta augmente.

Meme cible que graphe_eta_thm11.py : n=6, C_true = X0 AND X1 AND (NOT X2),
J+={0,1}, J-={2}, m=3.

Le bruit est retire independamment a CHAQUE epoch (pas fige une fois pour
toutes) pour que repeter les epochs sur la table de verite complete
equivaille a un echantillonnage i.i.d. de la loi bruitee. Pour PCL, on
reutilise leur fonction officielle PCL_convergence_all.PCL EXACTEMENT
(copie non modifiee de la logique de mise a jour), juste appelee un epoch
a la fois avec un tableau de labels fraichement bruite a chaque appel, en
chainant l'etat des automates entre les appels (mini-patch : la fonction
officielle accepte maintenant un ta_state initial optionnel, sinon
comportement identique).

Usage: python3 comparaison_recovery_bruit.py
"""
import itertools
import sys
import time

import numpy as np
from numba import njit

sys.path.insert(0, "../dpcl-classifier/Main_code_AAAI25")
from PCL_convergence_all import PCL as PCL_run_official  # noqa: E402

n, m = 6, 3
J_plus = [0, 1]
J_minus = [2]
S, a = 1.0, 0.3
states_pcl, p_pcl = 10000, 0.75
N_notre = 100


@njit(cache=True)
def notre_regle_bruit(n, epochs, X, Ystar, eta, N, S, a):
    state = np.random.choice(np.array([N, N + 1]), size=(n, 2)).astype(np.int64)
    n_ex = len(X)
    for _ in range(epochs):
        for e in range(n_ex):
            y = Ystar[e] ^ (1 if np.random.random() < eta else 0)
            for f in range(n):
                xf = X[e, f]
                v = state[f, 1]
                vbar = state[f, 0]
                if xf == 0:
                    if y == 0:
                        if np.random.random() < S and v < 2 * N:
                            v += 1
                        if np.random.random() < S and vbar > 1:
                            vbar -= 1
                    else:
                        if np.random.random() < a and v > 1:
                            v -= 1
                else:
                    if y == 0:
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


def formule_notre(state, N, n):
    return sorted(
        ([j + 1 for j in range(n) if state[j, 1] > N])
        + ([-j - 1 for j in range(n) if state[j, 0] > N])
    )


def formule_pcl(state, states, n):
    return sorted(
        ([j + 1 for j in range(n) if state[0, j, 1] > states])
        + ([-j - 1 for j in range(n) if state[0, j, 0] > states])
    )


def pcl_run_bruit(n, epochs, X, Ystar, eta, states, p, rng):
    """Reutilise PCL_run_official (code non modifie) tel quel : on ne
    peut pas chainer facilement son ta_state interne entre plusieurs
    appels (il est reinitialise a chaque appel), donc on lui fait faire
    UN SEUL appel avec epochs=1 sur une table concatenee de `epochs`
    copies de X, chacune avec son propre bruit tire independamment --
    strictement equivalent a `epochs` passes avec bruit frais a chaque
    fois, et ca ne modifie pas une seule ligne de leur logique de mise a
    jour."""
    Xe = np.tile(X, (epochs, 1))
    noise = (rng.random(epochs * len(X)) < eta).astype(np.int64)
    ye = np.tile(Ystar, epochs) ^ noise
    return PCL_run_official(n, 1, Xe, ye.astype(np.int64), 1, states, p)


def main():
    rng = np.random.default_rng(11)
    X = np.array(list(itertools.product([0, 1], repeat=n)), dtype=np.int64)
    Ystar = np.array([
        1 if all(row[j] == 1 for j in J_plus) and all(row[j] == 0 for j in J_minus) else 0
        for row in X
    ], dtype=np.int64)
    target = sorted([j + 1 for j in J_plus] + [-j - 1 for j in J_minus])

    etas = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.28, 0.30, 0.3125, 0.325, 0.35, 0.40]
    R, epochs = 100, 100

    # warm-up JIT
    notre_regle_bruit(n, 1, X, Ystar, 0.0, N_notre, S, a)
    pcl_run_bruit(n, 1, X, Ystar, 0.0, states_pcl, p_pcl, rng)

    print(f"Cible : {target}  (m={m})")
    print(f"{'eta':>8} {'PCL':>10} {'Notre regle':>12}")
    t0 = time.time()
    for eta in etas:
        succ_pcl = 0
        succ_notre = 0
        for _ in range(R):
            state_pcl = pcl_run_bruit(n, epochs, X, Ystar, eta, states_pcl, p_pcl, rng)
            if formule_pcl(state_pcl, states_pcl, n) == target:
                succ_pcl += 1
            state_notre = notre_regle_bruit(n, epochs, X, Ystar, eta, N_notre, S, a)
            if formule_notre(state_notre, N_notre, n) == target:
                succ_notre += 1
        print(f"{eta:>8.4f} {succ_pcl:>6}/{R:<3} {succ_notre:>8}/{R:<3}")
    print(f"temps total : {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
