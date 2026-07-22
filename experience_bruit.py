""" Experience 2 : robustesse au bruit (Theoreme 5, preuve_complete.tex:289).

Le theoreme dit : sous l'Hypothese H, si eta < mu/M (M=max(a+b,c+d), mu =
marge minimale sur tous les litteraux, calculee avec le label PROPRE),
alors la clause cible reste la configuration correcte malgre le bruit.

Protocole (aligne sur celui de PCL, protocole "a" -- table complete,
balayage cyclique epoch apres epoch) :
  - le bruit est retire INDEPENDAMMENT A CHAQUE PRESENTATION (pas fixe une
    fois pour toutes les epochs), en construisant une table repetee
    `epochs` fois avec un bruit different a chaque repetition ;
  - CETTE MEME table bruitee est partagee entre PCL et notre regle (pas
    deux tirages de bruit independants), pour une comparaison appariee ;
  - mu est calcule sur les frequences empiriques PROPRES (X, y sans
    bruit), comme le veut le theoreme ;
  - les resultats sont stratifies : eta < eta_max (garanti par la
    theorie) vs eta >= eta_max (hors garantie, stress test empirique).

Notes importantes :
  - eta_max=mu/M est le seuil garanti par LE THEOREME DE NOTRE REGLE. PCL
    est evalue sur le meme sous-ensemble d'essais (comparaison legitime),
    mais cette zone n'est theoriquement garantie que pour notre methode,
    pas pour PCL.
  - le succes a 100 epochs mesure un taux de recuperation en TEMPS FINI ;
    le theoreme garantit seulement que les signes des derives restent
    corrects (donc que la cible reste la configuration stationnairement
    preferee), pas qu'elle soit atteinte exactement apres 100 epochs.
  - le protocole reste le balayage cyclique standard de PCL (meme
    frequences que l'uniforme, mais pas un tirage i.i.d. avec remise
    comme le suppose formellement la preuve).
"""

import numpy as np

from experience_recuperation import generer_cible_et_donnees, formule_pcl, entrainer_clause


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
    M = max(a + b, c + d)
    ns = [4, 6, 8, 10, 12]
    etas = [0.05, 0.1, 0.2, 0.25, 0.30, 0.35, 0.40]

    rng = np.random.default_rng(0)
    for eta in etas:
        print(f"\n=== bruit eta={eta} (M={M}, eta_max=mu/{M}) ===")
        print(f"{'n':>3} {'PCL':>8} {'Notre regle':>12}   "
              f"{'--- zone garantie (pour notre regle) ---':>45}   "
              f"{'--- hors garantie ---':>25}   "
              f"{'H violee':>9}")
        print(f"{'':>3} {'':>8} {'':>12}   "
              f"{'PCL':>10} {'Notre regle':>12} {'# essais':>10}   "
              f"{'PCL':>10} {'Notre regle':>12} {'# essais':>10}")
        for n in ns:
            succ_pcl_total = succ_notre_total = 0
            succ_pcl_garanti = succ_notre_garanti = nb_garanti = 0
            succ_pcl_hors = succ_notre_hors = nb_hors = 0
            nb_hypothese_H_violee = 0
            total = nb_repetitions * runs
            for _ in range(nb_repetitions):
                for _ in range(runs):
                    target, X, y = generer_cible_et_donnees(n, rng)
                    mu = calculer_mu(X, y, target, a, b, c, d)
                    if mu <= 0:
                        nb_hypothese_H_violee += 1
                        eta_max = 0.0
                    else:
                        eta_max = mu / M
                    X_repete, y_bruite = construire_flux_bruite(X, y, epochs, eta, rng)
                    pred_pcl = formule_pcl_bruit(n, X_repete, y_bruite, states, p_pcl)
                    pred_notre = formule_notre_regle_bruit(n, X_repete, y_bruite, states, a, b, c, d)
                    ok_pcl = formule_correcte(pred_pcl, target)
                    ok_notre = formule_correcte(pred_notre, target)
                    succ_pcl_total += ok_pcl
                    succ_notre_total += ok_notre
                    if eta < eta_max:
                        nb_garanti += 1
                        succ_pcl_garanti += ok_pcl
                        succ_notre_garanti += ok_notre
                    else:
                        nb_hors += 1
                        succ_pcl_hors += ok_pcl
                        succ_notre_hors += ok_notre
            pct_pcl_garanti = 100 * succ_pcl_garanti / nb_garanti if nb_garanti else float("nan")
            pct_notre_garanti = 100 * succ_notre_garanti / nb_garanti if nb_garanti else float("nan")
            pct_pcl_hors = 100 * succ_pcl_hors / nb_hors if nb_hors else float("nan")
            pct_notre_hors = 100 * succ_notre_hors / nb_hors if nb_hors else float("nan")
            print(f"{n:>3} {100*succ_pcl_total/total:>7.1f} {100*succ_notre_total/total:>11.1f}   "
                  f"{pct_pcl_garanti:>10.1f} {pct_notre_garanti:>12.1f} {nb_garanti:>9}/{total}   "
                  f"{pct_pcl_hors:>10.1f} {pct_notre_hors:>12.1f} {nb_hors:>9}/{total}   "
                  f"{nb_hypothese_H_violee:>9}/{total}")


if __name__ == "__main__":
    main()
