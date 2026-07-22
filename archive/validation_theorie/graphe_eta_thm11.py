"""Graphe : validation empirique du Theoreme 11 (robustesse au bruit,
loi arbitraire) et de son corollaire exact (cas i.i.d. Bernoulli(1/2)).

Cible fixe : n=6 features, C_true = X0 AND X1 AND (NOT X2), features 3,4,5
non pertinentes (m=3 litteraux, J+={0,1}, J-={2}). Table de verite
complete (2^6=64 lignes, exactement Bernoulli(1/2) i.i.d. sur chaque
feature) -- exactement le cas du Corollaire "Bernoulli(1/2)".

Pour chaque niveau de bruit eta, on relabelle Y = Y* xor Bernoulli(eta)
(tire une fois par run), on entraine notre regle a 4 cas (une clause,
100 epochs sur la table complete), et on mesure le taux de recuperation
exacte de C_true sur R runs independants.

Deux seuils sont traces :
  - la borne generale (loi arbitraire, Theoreme 11) : mu/(S+a) -- conservatrice ;
  - le seuil exact (cas i.i.d. Bernoulli(1/2), Corollaire identif-bruit-bernoulli) :
    eta* = 2S/(4S+a*2^m).

Usage: python3 graphe_eta_thm11.py
"""
import itertools

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numba import njit

S, a, N = 1.0, 0.3, 100
n, m = 6, 3
J_plus = [0, 1]
J_minus = [2]


@njit(cache=True)
def entrainer(n, epochs, X, Ystar, eta, states, S, a):
    """Bruit de label re-tire independamment a CHAQUE visite d'un exemple
    (et non fige une fois pour toutes) -- necessaire pour que repeter les
    epochs sur la table de verite complete equivaille a un echantillonnage
    i.i.d. de la loi bruitee P(X,Y), comme suppose par le Theoreme 11."""
    state = np.random.choice(np.array([states, states + 1]), size=(n, 2)).astype(np.int64)
    for _ in range(epochs):
        for e in range(len(X)):
            y = Ystar[e] ^ (1 if np.random.random() < eta else 0)
            for f in range(n):
                xf = X[e, f]
                v = state[f, 1]
                vbar = state[f, 0]
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
                state[f, 1] = v
                state[f, 0] = vbar
    return state


def formule(state, states, n):
    return sorted(
        ([j + 1 for j in range(n) if state[j, 1] > states])
        + ([-j - 1 for j in range(n) if state[j, 0] > states])
    )


def cible_signee():
    t = []
    for j in J_plus:
        t.append(j + 1)
    for j in J_minus:
        t.append(-j - 1)
    return sorted(t)


def main():
    rng = np.random.default_rng(7)
    X = np.array(list(itertools.product([0, 1], repeat=n)), dtype=np.int64)
    Ystar = np.array([
        1 if all(row[j] == 1 for j in J_plus) and all(row[j] == 0 for j in J_minus) else 0
        for row in X
    ], dtype=np.int64)
    target = cible_signee()

    mu = 2 ** (-m) * min(S, a / 2)
    bound_generale = mu / (S + a)
    eta_star_exact = 2 * S / (4 * S + a * 2 ** m)
    print(f"borne generale mu/(S+a) = {bound_generale:.4f}")
    print(f"seuil exact eta* (Bernoulli 1/2) = {eta_star_exact:.4f}")

    etas = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.28, 0.30, 0.3125, 0.325, 0.35, 0.40, 0.45]
    R, epochs, states = 200, 100, 10000

    # warm-up JIT
    entrainer(n, 1, X, Ystar, 0.0, states, S, a)

    taux = []
    for eta in etas:
        succ = 0
        for _ in range(R):
            state = entrainer(n, epochs, X, Ystar, eta, states, S, a)
            if formule(state, states, n) == target:
                succ += 1
        taux.append(succ / R)
        print(f"eta={eta:.4f}  taux de recuperation exacte = {taux[-1]:.3f}")

    etas = np.array(etas)
    taux = np.array(taux)

    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    ax.plot(etas, taux, "o-", color="#1b6ca8", linewidth=2, markersize=6,
            label=r"$P(C_t \equiv C_{\mathrm{true}})$ empirique")
    ax.axvline(bound_generale, color="#c0392b", linestyle=":", linewidth=1.5,
               label=rf"borne générale $\mu/(S{{+}}a)={bound_generale:.3f}$ (Théorème 11)")
    ax.axvline(eta_star_exact, color="#2e8b57", linestyle="--", linewidth=1.5,
               label=rf"seuil exact $\eta^\star={eta_star_exact:.3f}$ (Bernoulli 1/2)")
    ax.set_xlabel(r"bruit de label $\eta$")
    ax.set_ylabel(r"taux de récupération exacte de $C_{\mathrm{true}}$")
    ax.set_title("Effet du bruit de label sur la récupération de la conjonction cible\n"
                 "(validation empirique du Théorème 11 et de son corollaire exact)")
    ax.legend(fontsize=8.5, loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("../figures/graphe_eta_thm11.png", dpi=150)
    print("Figure ecrite dans figures/graphe_eta_thm11.png")


if __name__ == "__main__":
    main()
