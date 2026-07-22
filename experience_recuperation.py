""" la regle a quatre cas (Definition 1), pour une seule clause.
"""

import sys

import numpy as np
from numba import njit

# Etape 3 : le vrai code officiel de PCL, importe tel quel (non modifie)
sys.path.insert(0, "dpcl-classifier/Main_code_AAAI25")
from PCL_convergence_all import PCL as PCL_run  # noqa: E402


@njit(cache=True)
def entrainer_clause(n, epochs, X, y, N, a, b, c, d):
    """Entraine une seule clause sur le jeu d'exemples (X, y).

    n      : nombre de features
    epochs : nombre de passages sur X (dans le meme ordre a chaque fois)
    X      : tableau (n_exemples, n) de 0/1
    y      : tableau (n_exemples,) de 0/1 (labels, sans bruit)
    N      : nombre d'etats par cote (2N etats au total par automate)
    a,b,c,d: parametres de la regle (Definition 1)

    Retourne l'etat final : tableau (n, 2)
        state[i, 1] = etat de v_i    (litteral x_i)
        state[i, 0] = etat de vbar_i (litteral not x_i)
    Initialisation aleatoire pres de la frontiere (N ou N+1).
    """
    state = np.random.choice(np.array([N, N + 1]), size=(n, 2)).astype(np.int64)
    for _ in range(epochs):
        for e in range(len(X)):
            yy = y[e]
            for i in range(n):
                xi = X[e, i]
                v = state[i, 1]
                vbar = state[i, 0]
                if xi == 0 and yy == 0:
                    if np.random.random() < a and v < 2 * N:
                        v += 1
                    if np.random.random() < c and vbar > 1:
                        vbar -= 1
                elif xi == 0 and yy == 1:
                    if np.random.random() < b and v > 1:
                        v -= 1
                    if np.random.random() < d and vbar < 2 * N:
                        vbar += 1
                elif xi == 1 and yy == 0:
                    if np.random.random() < c and v > 1:
                        v -= 1
                    if np.random.random() < a and vbar < 2 * N:
                        vbar += 1
                else:  # xi == 1 and yy == 1
                    if np.random.random() < d and v < 2 * N:
                        v += 1
                    if np.random.random() < b and vbar > 1:
                        vbar -= 1
                state[i, 1] = v
                state[i, 0] = vbar
    return state


# ---------------------------------------------------------------------------
# Etape 2 : generation de la cible et du jeu de donnees -- meme methode que
# le code officiel de PCL (dpcl_classifier/DPCL_ConvergenceTest.py) :
#   target = np.random.choice(3, n)   0=doit valoir 0, 1=doit valoir 1, 2=inutile
# avec la meme garde contre la cible entierement "inutile".
# ---------------------------------------------------------------------------
import itertools


def generer_cible_et_donnees(n, rng):
    """Tire une cible aleatoire et construit toute la table de
    verite (2^n lignes) avec les labels correspondants, sans bruit.

    Retourne :
        target_signe : liste triee des litteraux de la cible
                       (j+1 si x_j doit valoir 1, -(j+1) si x_j doit valoir 0)
        X : tableau (2^n, n) de 0/1 -- toutes les combinaisons possibles
        y : tableau (2^n,) de 0/1 -- le vrai label pour chaque ligne
    """
    target = rng.integers(0, 3, size=n)  # 0/1/2
    if np.all(target == 2):
        target[0] = 1  # garde contre la cible entierement inutile

    X = np.array(list(itertools.product([0, 1], repeat=n)), dtype=np.int64)
    y = np.ones(len(X), dtype=np.int64)
    for idx, x in enumerate(X):
        for j in range(n):
            if (x[j] == 1 and target[j] == 0) or (x[j] == 0 and target[j] == 1):
                y[idx] = 0
                break

    # Pas de sorted() : on garde l'ordre naturel des variables (comme le code
    # officiel de PCL), pour rester coherent avec formule_pcl/formule_notre_regle.
    target_signe = [j + 1 if target[j] == 1 else -(j + 1)
                     for j in range(n) if target[j] != 2]
    return target_signe, X, y



def formule_pcl(n, epochs, X, y, states, p):
    """Entraine 1 clause avec le vrai code de PCL, retourne les litteraux appris
    (meme ordre naturel, sans tri, que le code officiel)."""
    result = PCL_run(n, epochs, X, y, 1, states, p)  # 1 = une seule clause
    return [j + 1 if result[0, j, 1] > states else -(j + 1)
            for j in range(n) if result[0, j, 0] > states or result[0, j, 1] > states]


# ---------------------------------------------------------------------------
# Etape 4 : la boucle complete -- plusieurs runs, plusieurs valeurs de n,
# on compte les succes de chaque methode.
# ---------------------------------------------------------------------------

def formule_notre_regle(n, epochs, X, y, N, a, b, c, d):
    state = entrainer_clause(n, epochs, X, y, N, a, b, c, d)
    return [j + 1 if state[j, 1] > N else -(j + 1)
            for j in range(n) if state[j, 0] > N or state[j, 1] > N]


def main():
    states, epochs, runs = 10000, 100, 100
    nb_repetitions = 10  # comme PCL : chaque score est la moyenne de 10 runs de 100 essais
    p_pcl = 0.95  # PCL-a (papier : table complete)
    a, b, c, d = 1.0, 0.3, 1.0, 0.1  # a,b,c,d in (0,1], a<=c et d<=b (Hypothese H)
    ns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    rng = np.random.default_rng(0)
    print(f"{'n':>3} {'PCL':>10} {'Notre regle':>14}   (moyenne sur {nb_repetitions} runs de {runs} essais)")
    for n in ns:
        scores_pcl = []
        scores_notre = []
        for _ in range(nb_repetitions):
            succ_pcl = succ_notre = 0
            for _ in range(runs):
                target, X, y = generer_cible_et_donnees(n, rng)
                if formule_pcl(n, epochs, X, y, states, p_pcl) == target:
                    succ_pcl += 1
                if formule_notre_regle(n, epochs, X, y, states, a, b, c, d) == target:
                    succ_notre += 1
            scores_pcl.append(succ_pcl)
            scores_notre.append(succ_notre)
        print(f"{n:>3} {np.mean(scores_pcl):>10.1f} {np.mean(scores_notre):>14.1f}"
              f"   (scores : PCL={scores_pcl}  notre_regle={scores_notre})")


if __name__ == "__main__":
    main()
