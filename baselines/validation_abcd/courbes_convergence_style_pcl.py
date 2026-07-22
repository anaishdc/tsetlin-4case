"""Reproduit, avec notre regle generalisee a 4 parametres libres (a,b,c,d)
(1 clause, sans conditions ni boosting), les deux courbes de validation que
PCL utilise pour verifier sa propre theorie (Figures 3a et 3b du papier
AAAI-25) -- pas une comparaison contre PCL/TM (deja faite dans
comparaison_recovery.py), mais le meme type de courbe de convergence, pour
verifier que notre regle generalisee a le meme genre de comportement
documente et comparable.

Experience 1 (mirroir Figure 3a) : (a,b,c,d) fixes, n in {2,4,6,8}, succes
moyen (sur R runs, cible aleatoire a chaque run, table de verite complete a
chaque epoch) en fonction du nombre d'epochs.

Experience 2 (mirroir Figure 3b) : n=4 fixe, a=c=1.0 et b fixes, d variable
(le degre de liberte nouveau de preuve_generale_abcd.tex) -- pour verifier
qu'on retrouve la meme forme qualitative que PCL (au-dela du seuil d=b,
effondrement), mais avec notre propre parametre d au lieu de leur p.

Usage: python3 courbes_convergence_style_pcl.py
"""
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numba import njit

from comparaison_recovery import generer_cible_et_table  # reutilise le meme generateur de cible


@njit(cache=True)
def entrainer_clause_abcd(examples, labels, epochs, N, a, b, c, d):
    """Regle a 4 parametres libres (Definition 1 de preuve_generale_abcd.tex),
    meme protocole que PCL (table de verite complete a chaque epoch)."""
    n = examples.shape[1]
    state = np.random.choice(np.array([N, N + 1]), size=(n, 2)).astype(np.int64)
    for _ in range(epochs):
        for e in range(len(examples)):
            yy = labels[e]
            for f in range(n):
                xf = examples[e, f]
                v = state[f, 1]
                vbar = state[f, 0]
                if xf == 0 and yy == 0:
                    if np.random.random() < a and v < 2 * N:
                        v += 1
                    if np.random.random() < c and vbar > 1:
                        vbar -= 1
                elif xf == 0 and yy == 1:
                    if np.random.random() < b and v > 1:
                        v -= 1
                    if np.random.random() < d and vbar < 2 * N:
                        vbar += 1
                elif xf == 1 and yy == 0:
                    if np.random.random() < c and v > 1:
                        v -= 1
                    if np.random.random() < a and vbar < 2 * N:
                        vbar += 1
                else:
                    if np.random.random() < d and v < 2 * N:
                        v += 1
                    if np.random.random() < b and vbar > 1:
                        vbar -= 1
                state[f, 1] = v
                state[f, 0] = vbar
    return state


def formule(state, N, n):
    return sorted([j + 1 if state[j, 1] > N else -j - 1
                   for j in range(n) if state[j, 1] > N or state[j, 0] > N])


def taux_succes(n, epochs, N, a, b, c, d, R, rng):
    succ = 0
    for _ in range(R):
        _, examples, labels, target = generer_cible_et_table(n, rng)
        state = entrainer_clause_abcd(examples, labels, epochs, N, a, b, c, d)
        if formule(state, N, n) == sorted(target):
            succ += 1
    return 100.0 * succ / R


def experience1():
    """Mirroir Figure 3a : (a,b,c,d) fixes, n variable, succes vs epochs."""
    rng = np.random.default_rng(1)
    N = 2000  # assez grand pour que n=8 sature dans la plage d'epochs testee
    a, b, c, d = 1.0, 0.3, 1.0, 0.0  # a=c (Hypothese H a l'egalite), d<b
    R = 30
    ns = [2, 4, 6, 8]
    epochs_list = [0, 25, 50, 100, 200, 300, 400, 600, 800, 1000]

    entrainer_clause_abcd(np.zeros((4, 2), dtype=np.int64), np.zeros(4, dtype=np.int64), 1, N, a, b, c, d)  # warm-up

    fig, ax = plt.subplots(figsize=(6.5, 5))
    print(f"=== Experience 1 (mirroir Fig. 3a PCL) : a={a}, b={b}, c={c}, d={d} ===")
    for n in ns:
        rates = []
        for ep in epochs_list:
            if ep == 0:
                rates.append(0.0)
                continue
            rates.append(taux_succes(n, ep, N, a, b, c, d, R, rng))
        print(f"n={n}: " + " ".join(f"{r:.0f}" for r in rates))
        ax.plot(epochs_list, rates, "o-", label=f"n={n}", linewidth=2, markersize=5)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Succes moyen (%)")
    ax.set_title(f"Notre regle (a,b,c,d) -- succes vs epochs\n(a={a},b={b},c={c},d={d}, {R} runs/point, mirroir Fig. 3a de PCL)")
    ax.legend(title="n")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("../../figures/courbe_convergence_notre_regle_n.png", dpi=150)
    print("Figure ecrite : figures/courbe_convergence_notre_regle_n.png")


def experience2():
    """Mirroir Figure 3b : n=4 fixe, a=c,b fixes, d variable (le nouveau
    degre de liberte), succes vs epochs."""
    rng = np.random.default_rng(2)
    N = 200
    n = 4
    a, c, b = 1.0, 1.0, 0.5  # seuil theorique attendu : d < b = 0.5
    R = 30
    ds = [0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0]
    epochs_list = [0, 20, 40, 60, 80, 100, 150, 200, 300, 500]

    fig, ax = plt.subplots(figsize=(6.5, 5))
    print(f"\n=== Experience 2 (mirroir Fig. 3b PCL) : n={n}, a=c={a}, b={b} (seuil d<{b}) ===")
    for d in ds:
        rates = []
        for ep in epochs_list:
            if ep == 0:
                rates.append(0.0)
                continue
            rates.append(taux_succes(n, ep, N, a, b, c, d, R, rng))
        print(f"d={d}: " + " ".join(f"{r:.0f}" for r in rates))
        ax.plot(epochs_list, rates, "o-", label=f"d={d}", linewidth=1.8, markersize=4)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Succes moyen (%)")
    ax.set_title(f"Notre regle (a,b,c,d) -- succes vs epochs, n={n} fixe\n(a=c={a}, b={b}, seuil theorique d<{b}, {R} runs/point, mirroir Fig. 3b de PCL)")
    ax.legend(title="d", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("../../figures/courbe_convergence_notre_regle_d.png", dpi=150)
    print("Figure ecrite : figures/courbe_convergence_notre_regle_d.png")


if __name__ == "__main__":
    t0 = time.time()
    experience1()
    experience2()
    print(f"\ntemps total : {time.time()-t0:.1f}s")
