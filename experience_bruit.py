""" Experience 2 : robustesse au bruit.

Meme protocole que experience_recuperation.py (Exp 1), avec un seul
ajout : chaque label est bruite (invers'e avec probabilite eta)
INDEPENDAMMENT A CHAQUE PRESENTATION (pas une fois pour toutes les
epochs) -- en construisant une table repetee `epochs` fois avec un bruit
different a chaque repetition. CETTE MEME table bruitee est partagee
entre PCL et notre regle (comparaison appariee, pas deux tirages de
bruit independants). L'evaluation se fait toujours contre la vraie
cible propre (non bruitee).

Pour PCL, on reutilise leur vrai code tel quel (formule_pcl), en lui
donnant la table repetee/bruitee avec epochs=1 -- equivalent a rebruiter
a chaque epoch, sans modifier leur code.
"""

import numpy as np

from experience_recuperation import generer_cible_et_donnees, formule_pcl, entrainer_clause


def construire_flux_bruite(X, y, epochs, eta, rng):
    """Repete la table `epochs` fois, bruite chaque copie independamment
    (bruit frais a chaque epoch). Retourne le MEME flux pour etre partage
    entre PCL et notre regle (comparaison appariee, meme realisation de
    bruit pour les deux)."""
    X_repete = np.tile(X, (epochs, 1))
    y_repete = np.tile(y, epochs)
    inverser = rng.random(len(y_repete)) < eta
    y_bruite = np.where(inverser, 1 - y_repete, y_repete)
    return X_repete, y_bruite


def formule_pcl_bruit(n, X_repete, y_bruite, states, p):
    """Reutilise formule_pcl (code officiel de PCL) tel quel, sur le flux
    deja repete/bruite (epochs=1 -- un seul passage sur le flux complet)."""
    return formule_pcl(n, 1, X_repete, y_bruite, states, p)


def formule_notre_regle_bruit(n, X_repete, y_bruite, N, a, b, c, d):
    """Reutilise entrainer_clause tel quel, sur le MEME flux deja
    repete/bruite que PCL (comparaison appariee). Retourne None si la
    clause apprise est contradictoire (x_j et ~x_j inclus en meme temps)."""
    state = entrainer_clause(n, 1, X_repete, y_bruite, N, a, b, c, d)
    formule = []
    for j in range(n):
        pos_inclus = state[j, 1] > N
        neg_inclus = state[j, 0] > N
        if pos_inclus and neg_inclus:
            return None
        if pos_inclus:
            formule.append(j + 1)
        elif neg_inclus:
            formule.append(-(j + 1))
    return formule


def formule_correcte(formule, target):
    """Compare par ensemble, pas par ordre de liste -- insensible a un
    eventuel ordre different entre target/formule_pcl/notre_regle."""
    return formule is not None and set(formule) == set(target)


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
                    X_repete, y_bruite = construire_flux_bruite(X, y, epochs, eta, rng)
                    pred_pcl = formule_pcl_bruit(n, X_repete, y_bruite, states, p_pcl)
                    pred_notre = formule_notre_regle_bruit(n, X_repete, y_bruite, states, a, b, c, d)
                    succ_pcl += formule_correcte(pred_pcl, target)
                    succ_notre += formule_correcte(pred_notre, target)
                scores_pcl.append(succ_pcl)
                scores_notre.append(succ_notre)
            print(f"{n:>3} {np.mean(scores_pcl):>10.1f} {np.mean(scores_notre):>14.1f}"
                  f"   (scores : PCL={scores_pcl}  notre_regle={scores_notre})")


if __name__ == "__main__":
    main()
