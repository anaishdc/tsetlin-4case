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


def verifier_signes_bruit(X, y, target, a, b, c, d, eta):
    """VERIFICATION DIRECTE du Theoreme 5, sans aucun entrainement.

    Recalcule les derives Delta_j, Delta_bar_j EXACTEMENT (calcul
    analytique, pas d'echantillonnage) sous le canal de bruit symetrique
    de taux eta (chaque label invers'e independamment avec probabilite
    eta), et verifie que leurs signes sont ceux requis pour que la
    cible reste la configuration correcte (Theoreme 4). C'est un test
    purement mathematique de ce que le theoreme affirme -- pas une
    consequence empirique d'un entrainement fini.

    Frequences bruitees, calculees exactement a partir des frequences
    propres A,B,C,D (puisque le bruit est un canal symetrique connu) :
        A_bruite = A*(1-eta) + C*eta
        B_bruite = B*(1-eta) + D*eta
        C_bruite = C*(1-eta) + A*eta
        D_bruite = D*(1-eta) + B*eta
    """
    target_set = set(target)
    for j in range(X.shape[1]):
        xj = X[:, j]
        A = np.mean((xj == 0) & (y == 0))
        B = np.mean((xj == 1) & (y == 0))
        C = np.mean((xj == 0) & (y == 1))
        D = np.mean((xj == 1) & (y == 1))
        A_b = A * (1 - eta) + C * eta
        B_b = B * (1 - eta) + D * eta
        C_b = C * (1 - eta) + A * eta
        D_b = D * (1 - eta) + B * eta
        delta = a * A_b + d * D_b - b * C_b - c * B_b
        delta_bar = d * C_b + a * B_b - c * A_b - b * D_b
        if (j + 1) in target_set:
            if not (delta > 0 and delta_bar < 0):
                return False
        elif -(j + 1) in target_set:
            if not (delta < 0 and delta_bar > 0):
                return False
        else:
            if not (delta < 0 and delta_bar < 0):
                return False
    return True


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
    etas = [0.05, 0.1, 0.2, 0.25, 0.30, 0.35, 0.40]

    for eta in etas:
        print(f"\n=== bruit eta={eta} (M={M}) ===")
        print(f"{'n':>3}   {'PREUVE DIRECTE (signes)':>24} {'# essais':>10}   "
              f"{'consequence pratique (100 epochs)':>34} {'# essais':>10}")
        for n in ns:
            rng = np.random.default_rng(0)
            nb_H_violee = 0
            signes_ok = nb_garanti = 0
            succ_garanti = 0
            for _ in range(essais):
                target, X, y = generer_cible_et_donnees(n, rng)
                mu = calculer_mu(X, y, target, a, b, c, d)
                if mu <= 0:
                    nb_H_violee += 1
                    continue
                eta_max = mu / M
                if eta >= eta_max:
                    continue
                nb_garanti += 1
                # (A) preuve directe : les signes des derives bruitees
                # sont-ils corrects ? Calcul analytique, garanti par le
                # theoreme, ne depend d'aucun entrainement.
                signes_ok += verifier_signes_bruit(X, y, target, a, b, c, d, eta)
                # (B) consequence pratique : est-ce que 100 epochs
                # d'entrainement suffisent a retrouver exactement la
                # clause ? Question separee (temps fini), pas ce que le
                # theoreme garantit directement.
                X_repete, y_bruite = construire_flux_bruite(X, y, epochs, eta, rng)
                pred = formule_notre_regle_bruit(n, X_repete, y_bruite, states, a, b, c, d)
                succ_garanti += formule_correcte(pred, target)
            pct_signes = 100 * signes_ok / nb_garanti if nb_garanti else float("nan")
            pct_pratique = 100 * succ_garanti / nb_garanti if nb_garanti else float("nan")
            print(f"{n:>3}   {pct_signes:>23.1f}% {nb_garanti:>9}/{essais}   "
                  f"{pct_pratique:>33.1f}% {nb_garanti:>9}/{essais}"
                  f"   (H violee: {nb_H_violee}/{essais})")


if __name__ == "__main__":
    main()
