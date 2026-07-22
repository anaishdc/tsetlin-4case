""" Experience 2 : validation empirique du Theoreme 5 (robustesse au bruit).

Le theoreme dit : sous l'Hypothese H, si eta < mu/M (M=max(a+b,c+d), mu =
marge minimale sur tous les litteraux, calculee avec le label PROPRE),
alors la clause cible reste la configuration correcte malgre le bruit.

But de cette experience : verifier directement cette prediction, PAS
comparer a PCL (PCL n'a pas d'equivalent a ce theoreme, la comparaison
n'a pas de sens ici).

Protocole :
  - pour chaque cible tiree, on calcule mu (avec X, y PROPRES) et donc
    eta_max = mu/M ;
  - le bruit est retire INDEPENDAMMENT A CHAQUE PRESENTATION (pas fixe
    une fois pour toutes les epochs), en construisant une table repetee
    `epochs` fois avec un bruit different a chaque repetition ;
  - on entraine notre regle sur cette table bruitee, et on verifie si la
    clause apprise correspond exactement a la cible propre ;
  - les resultats sont stratifies : eta < eta_max (garanti par la
    theorie) vs eta >= eta_max (hors garantie, stress test empirique).
    Le theoreme predit 100% de succes dans la premiere categorie.
"""

import numpy as np

from experience_recuperation import generer_cible_et_donnees, entrainer_clause


def calculer_mu(X, y, target, a, b, c, d):
    """Marge minimale (mu du Theoreme 5) sur tous les litteraux, calculee
    a partir des frequences empiriques (X, y) SANS bruit."""
    target_set = set(target)
    marges = []
    for j in range(X.shape[1]):
        xj = X[:, j]
        A = np.mean((xj == 0) & (y == 0))
        B = np.mean((xj == 1) & (y == 0))
        C = np.mean((xj == 0) & (y == 1))
        D = np.mean((xj == 1) & (y == 1))
        delta = a * A + d * D - b * C - c * B
        delta_bar = d * C + a * B - c * A - b * D
        if (j + 1) in target_set:
            marges.extend([delta, -delta_bar])
        elif -(j + 1) in target_set:
            marges.extend([-delta, delta_bar])
        else:
            marges.extend([-delta, -delta_bar])
    return min(marges)


def construire_flux_bruite(X, y, epochs, eta, rng):
    """Repete la table `epochs` fois, bruite chaque copie independamment
    (bruit frais a chaque epoch)."""
    X_repete = np.tile(X, (epochs, 1))
    y_repete = np.tile(y, epochs)
    inverser = rng.random(len(y_repete)) < eta
    y_bruite = np.where(inverser, 1 - y_repete, y_repete)
    return X_repete, y_bruite


def formule_notre_regle_bruit(n, X_repete, y_bruite, N, a, b, c, d):
    """Reutilise entrainer_clause tel quel, sur le flux repete/bruite.
    Retourne None si la clause apprise est contradictoire (x_j et ~x_j
    inclus en meme temps)."""
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
    eventuel ordre different entre target et la formule apprise."""
    return formule is not None and set(formule) == set(target)


def main():
    states, epochs, essais = 10000, 100, 2000
    a, b, c, d = 1.0, 0.3, 1.0, 0.1
    M = max(a + b, c + d)
    ns = [4, 6, 8, 10, 12]
    # etas petits pour avoir des essais dans la zone garantie meme a grand n
    # (mu/M retrecit vite avec n), + quelques valeurs plus grandes pour le
    # stress test hors garantie.
    etas = [0.005, 0.01, 0.02, 0.05, 0.1]

    for eta in etas:
        print(f"\n=== bruit eta={eta} (M={M}) ===")
        print(f"{'n':>3}   {'zone garantie: succes':>22} {'# essais':>10}   "
              f"{'hors garantie: succes':>22} {'# essais':>10}")
        for n in ns:
            rng = np.random.default_rng(0)
            succ_garanti = nb_garanti = 0
            succ_hors = nb_hors = 0
            for _ in range(essais):
                target, X, y = generer_cible_et_donnees(n, rng)
                mu = calculer_mu(X, y, target, a, b, c, d)
                eta_max = mu / M if mu > 0 else 0.0
                X_repete, y_bruite = construire_flux_bruite(X, y, epochs, eta, rng)
                pred = formule_notre_regle_bruit(n, X_repete, y_bruite, states, a, b, c, d)
                ok = formule_correcte(pred, target)
                if eta < eta_max:
                    nb_garanti += 1
                    succ_garanti += ok
                else:
                    nb_hors += 1
                    succ_hors += ok
            pct_garanti = 100 * succ_garanti / nb_garanti if nb_garanti else float("nan")
            pct_hors = 100 * succ_hors / nb_hors if nb_hors else float("nan")
            print(f"{n:>3}   {pct_garanti:>21.1f}% {nb_garanti:>9}/{essais}   "
                  f"{pct_hors:>21.1f}% {nb_hors:>9}/{essais}")


if __name__ == "__main__":
    main()
