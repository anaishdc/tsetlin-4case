""" Experience 2 : robustesse au bruit (Theoreme 5).

Meme protocole que experience_recuperation.py (Exp 1), avec un seul
ajout : les labels d'entrainement sont bruites (chaque label invers,e
independamment avec probabilite eta) avant l'entrainement. L'evaluation
se fait toujours contre la vraie cible propre (non bruitee).
"""

import numpy as np

from experience_recuperation import (
    generer_cible_et_donnees,
    formule_pcl,
    formule_notre_regle,
)


def bruiter_labels(y, eta, rng):
    """Inverse chaque label independamment avec probabilite eta."""
    inverser = rng.random(len(y)) < eta
    return np.where(inverser, 1 - y, y)


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
                    y_bruite = bruiter_labels(y, eta, rng)
                    if formule_pcl(n, epochs, X, y_bruite, states, p_pcl) == target:
                        succ_pcl += 1
                    if formule_notre_regle(n, epochs, X, y_bruite, states, a, b, c, d) == target:
                        succ_notre += 1
                scores_pcl.append(succ_pcl)
                scores_notre.append(succ_notre)
            print(f"{n:>3} {np.mean(scores_pcl):>10.1f} {np.mean(scores_notre):>14.1f}"
                  f"   (scores : PCL={scores_pcl}  notre_regle={scores_notre})")


if __name__ == "__main__":
    main()
