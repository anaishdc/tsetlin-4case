""" Graphe : validation du Theoreme 5 (robustesse au bruit).

Taux de succes (recuperation EXACTE de la cible, apres 100 epochs
d'entrainement) en fonction de r = eta/eta_max, ou eta_max = mu/M est le
seuil garanti par le Theoreme 5 pour la cible tiree.

r < 1 : zone garantie par la theorie (le theoreme promet le succes)
r > 1 : hors garantie (pas de promesse, simple stress test empirique)

n est mélangé (4,6,8,10,12) a chaque r puisque r normalise deja par le
seuil propre a chaque cible -- une seule courbe suffit.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experience_recuperation import generer_cible_et_donnees
from experience_bruit import calculer_mu, construire_flux_bruite, formule_notre_regle_bruit, formule_correcte

states, epochs = 3000, 100
a, b, c, d = 1.0, 0.3, 1.0, 0.1
M = max(a + b, c + d)
ns = [4, 6, 8, 10, 12]
rs = [0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
essais_par_r = 200


def main():
    rng = np.random.default_rng(0)
    taux_succes = []
    for r in rs:
        succ = total = 0
        for i in range(essais_par_r):
            n = ns[i % len(ns)]
            target, X, y = generer_cible_et_donnees(n, rng)
            mu = calculer_mu(X, y, target, a, b, c, d)
            if mu <= 0:
                continue  # Hypothese H violee pour cette cible, non identifiable meme sans bruit
            eta_max = mu / M
            eta = min(r * eta_max, 0.5)  # plafonne : au-dela, le bruit devient degenere (>=50%)
            X_repete, y_bruite = construire_flux_bruite(X, y, epochs, eta, rng)
            pred = formule_notre_regle_bruit(n, X_repete, y_bruite, states, a, b, c, d)
            succ += formule_correcte(pred, target)
            total += 1
        taux = 100 * succ / total if total else float("nan")
        taux_succes.append(taux)
        print(f"r={r:>4}: {succ:>3}/{total:>3} = {taux:.1f}%")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rs, taux_succes, marker="o", linewidth=2)
    ax.axvline(1.0, color="red", linestyle="--", label=r"$r=1$ (seuil du théorème 5)")
    ax.set_xscale("log")
    ax.set_xlabel(r"$r = \eta / \eta_{\max}$ (échelle log)")
    ax.set_ylabel("taux de succès (%)")
    ax.set_title("Récupération exacte vs bruit relatif au seuil garanti")
    ax.set_ylim(-5, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("figures/succes_vs_r_theoreme5.png", dpi=150)
    print("\n-> figures/succes_vs_r_theoreme5.png")


if __name__ == "__main__":
    main()
