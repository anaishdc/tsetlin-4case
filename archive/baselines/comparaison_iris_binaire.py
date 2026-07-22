"""Comparaison PCL / notre regle sur le protocole Iris binaire exact de PCL
(PCL_BinaryIRIS.py, AAAI-25) : classe 1 (versicolor) vs reste, 16 features
binaires, split 70/30 ALEATOIRE (shuffle puis coupe -- pas sequentiel,
contrairement a ce qui avait ete suppose au depart), p_i~U(0.6,0.8) par
clause, epochs=100, clauses=16. PCL est appele via une copie litterale de
leurs fonctions (pcl_iris_officiel.py), pas une reimplementation, pour
eviter toute erreur de retranscription sur une regle plus subtile que celle
de PCL_convergence_all.py (les deux automates d'une meme feature sont ici
mis a jour de façon stochastique des deux cotes, y compris sur exemple
negatif).

Notre regle est reduite a sa forme la plus simple pour cette comparaison :
M clauses independantes sur les 16 features (pas de conditions aleatoires,
pas d'AdaBoost -- ces mecanismes sont evalues separement en Section 6),
vote par disjonction (une clause suffit), memes hyperparametres que le
reste du papier (S=1.0, a=0.3).

Usage: python3 comparaison_iris_binaire.py
"""
import time

import numpy as np
from numba import njit

from pcl_iris_officiel import PCL as pcl_fit_officiel
from pcl_iris_officiel import Accuracy as accuracy_officiel

DATA_PATH = "dpcl-classifier/dpcl_classifier/binary_iris.txt"


@njit(cache=True)
def notre_regle_fit(n, epochs, X, Y, clauses, states, S, a):
    ta_state = np.random.choice(np.array([states, states + 1]),
                                 size=(clauses, n, 2)).astype(np.int32)
    for _ in range(epochs):
        for e in range(len(X)):
            y = Y[e]
            for f in range(n):
                xf = X[e, f]
                for j in range(clauses):
                    v = ta_state[j, f, 1]
                    vbar = ta_state[j, f, 0]
                    if xf == 0:
                        if y == 0:
                            if np.random.random() < S and v < 2 * states:
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
                            if np.random.random() < S and vbar < 2 * states:
                                vbar += 1
                        else:
                            if np.random.random() < a and vbar > 1:
                                vbar -= 1
                    ta_state[j, f, 1] = v
                    ta_state[j, f, 0] = vbar
    return ta_state


def formulas_from_state(ta_state, states, n, clauses):
    formulas = []
    for i in range(clauses):
        c = []
        for j in range(n):
            if ta_state[i, j, 1] > states:
                c.append(j + 1)
            if ta_state[i, j, 0] > states:
                c.append(-j - 1)
        formulas.append(c)
    return formulas


def main():
    data = np.loadtxt(DATA_PATH).astype(np.int32)
    states, epochs, clauses, runs = 10000, 100, 16, 200
    pl, pu = 0.6, 0.8
    S, a = 1.0, 0.3

    accs_pcl = np.empty(runs)
    accs_notre = np.empty(runs)

    # warm-up JIT
    probs0 = np.random.uniform(pl, pu, clauses)
    y0 = (data[:10, 16] == 1).astype(np.int32)
    pcl_fit_officiel(16, 1, data[:10, :16], y0, clauses, states, probs0)
    notre_regle_fit(16, 1, data[:10, :16], y0, clauses, states, S, a)

    t0 = time.time()
    for r in range(runs):
        rng = np.random.default_rng(1000 + r)
        perm = rng.permutation(len(data))
        d = data[perm]
        n_train = int(0.7 * len(d))
        X_train = d[:n_train, :16]
        Y_train = (d[:n_train, 16] == 1).astype(np.int32)
        X_test = d[n_train:, :16]
        Y_test = (d[n_train:, 16] == 1).astype(np.int32)

        probs = np.random.uniform(pl, pu, clauses)
        state_pcl = pcl_fit_officiel(16, epochs, X_train, Y_train, clauses, states, probs)
        f_pcl = formulas_from_state(state_pcl, states, 16, clauses)
        accs_pcl[r] = accuracy_officiel(X_test, f_pcl, 16, Y_test)

        state_notre = notre_regle_fit(16, epochs, X_train, Y_train, clauses, states, S, a)
        f_notre = formulas_from_state(state_notre, states, 16, clauses)
        accs_notre[r] = accuracy_officiel(X_test, f_notre, 16, Y_test)

    dt = time.time() - t0
    print(f"Iris binaire (versicolor vs reste), {runs} runs, {clauses} clauses, {epochs} epochs, split 70/30 aleatoire")
    print(f"PCL (code officiel) : {100*accs_pcl.mean():.2f}% +/- {100*accs_pcl.std():.2f}%  "
          f"(min={100*accs_pcl.min():.2f}%, max={100*accs_pcl.max():.2f}%)")
    print(f"Notre regle          : {100*accs_notre.mean():.2f}% +/- {100*accs_notre.std():.2f}%  "
          f"(min={100*accs_notre.min():.2f}%, max={100*accs_notre.max():.2f}%)")
    print(f"temps total : {dt:.1f}s")


if __name__ == "__main__":
    main()
