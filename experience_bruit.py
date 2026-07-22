""" Experience 2 : robustesse au bruit (Theoreme 5).

Meme protocole que experience_recuperation.py (Exp 1), avec un seul
ajout : chaque label est bruite (invers'e avec probabilite eta)
INDEPENDAMMENT A CHAQUE PRESENTATION (pas une fois pour toutes les
epochs) -- c'est le modele suppose par la theorie (chaque retour de
l'environnement est bruite independamment). L'evaluation se fait
toujours contre la vraie cible propre (non bruitee).

Pour notre regle, le bruit est tire directement dans la boucle
d'entrainement (nouvelle fonction). Pour PCL, on reutilise leur vrai
code tel quel (formule_pcl), en repetant la table `epochs` fois avec
un bruit different a chaque repetition -- equivalent a rebruiter a
chaque epoch, sans toucher a leur code.
"""

import numpy as np
from numba import njit

from experience_recuperation import generer_cible_et_donnees, formule_pcl


@njit(cache=True)
def entrainer_clause_bruit(n, epochs, X, y, N, a, b, c, d, eta):
    """Comme entrainer_clause (experience_recuperation.py), mais le label
    est bruite (invers'e avec probabilite eta) independamment a chaque
    presentation d'exemple."""
    state = np.random.choice(np.array([N, N + 1]), size=(n, 2)).astype(np.int64)
    for _ in range(epochs):
        for e in range(len(X)):
            yy = y[e]
            if np.random.random() < eta:
                yy = 1 - yy
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


def formule_notre_regle_bruit(n, epochs, X, y, N, a, b, c, d, eta):
    state = entrainer_clause_bruit(n, epochs, X, y, N, a, b, c, d, eta)
    return [j + 1 if state[j, 1] > N else -(j + 1)
            for j in range(n) if state[j, 0] > N or state[j, 1] > N]


def formule_pcl_bruit(n, epochs, X, y, states, p, eta, rng):
    """Reutilise formule_pcl (code officiel de PCL) tel quel : construit
    une table repetee `epochs` fois, avec un bruit different (frais) a
    chaque repetition, puis appelle PCL avec epochs=1 sur cette table --
    equivalent a rebruiter a chaque epoch, sans modifier leur code."""
    X_repete = np.tile(X, (epochs, 1))
    y_repete = np.tile(y, epochs)
    inverser = rng.random(len(y_repete)) < eta
    y_bruite = np.where(inverser, 1 - y_repete, y_repete)
    return formule_pcl(n, 1, X_repete, y_bruite, states, p)


def main():
    states, epochs, runs = 10000, 100, 100
    nb_repetitions = 10
    p_pcl = 0.95
    a, b, c, d = 1.0, 0.3, 1.0, 0.1
    ns = [4, 6, 8, 10, 12]
    etas = [0.05, 0.1, 0.2, 0.25, 0.30, 0.35, 0.40]

    rng = np.random.default_rng(0)
    for eta in etas:
        print(f"\n=== bruit eta={eta} ===")
        print(f"{'n':>3} {'PCL':>10} {'Notre regle':>14}   (moyenne sur {nb_repetitions} runs de {runs} essais)")
        for n in ns:
            scores_pcl = []
            scores_notre = []
            for _ in range(nb_repetitions):
                succ_pcl = succ_notre = 0
                for _ in range(runs):
                    target, X, y = generer_cible_et_donnees(n, rng)
                    if formule_pcl_bruit(n, epochs, X, y, states, p_pcl, eta, rng) == target:
                        succ_pcl += 1
                    if formule_notre_regle_bruit(n, epochs, X, y, states, a, b, c, d, eta) == target:
                        succ_notre += 1
                scores_pcl.append(succ_pcl)
                scores_notre.append(succ_notre)
            print(f"{n:>3} {np.mean(scores_pcl):>10.1f} {np.mean(scores_notre):>14.1f}"
                  f"   (scores : PCL={scores_pcl}  notre_regle={scores_notre})")


if __name__ == "__main__":
    main()
