""" Figure Exp 2 (validation du Theoreme 5 -- robustesse au bruit) : axe
X = bruit reel eta (%), ligne = seuil theorique eta<=mu/M, pour UNE
cible concrete fixee (n=4, cible=[1,2,3,4], a=1.0 b=0.3 c=1.0 d=0.1).

Version "papier" complete suite a une revue : intervalles de confiance
binomiaux (Wilson, 95%), seuil clarifie (regime garanti, pas transition
exacte), legende/polices allegees, axe en %, zone garantie neutre.
"""

import math

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.labelsize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "legend.frameon": False,
})
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, ScalarFormatter, NullFormatter


def wilson_ci(k, n, z=1.96):
    """Intervalle de confiance binomial de Wilson (95% par defaut)."""
    if n == 0:
        return 0.0, 0.0, 0.0
    phat = k / n
    denom = 1 + z**2 / n
    centre = (phat + z**2 / (2 * n)) / denom
    marge = (z / denom) * math.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))
    return phat, max(0.0, centre - marge), min(1.0, centre + marge)


eta_max = 0.05288
etas = [0.01058, 0.02644, 0.05288, 0.10577, 0.15865, 0.21154, 0.26442, 0.31731]
n_essais = 40
k_succes = [40, 40, 40, 40, 40, 24, 8, 0]

etas_pct = [e * 100 for e in etas]
eta_max_pct = eta_max * 100

taux, bas, haut = [], [], []
for k in k_succes:
    phat, lo, hi = wilson_ci(k, n_essais)
    taux.append(100 * phat)
    bas.append(100 * (phat - lo))
    haut.append(100 * (hi - phat))

BLUE = "#1f5fa8"
RED = "#c0392b"
ZONE = "#4d78ad"

fig, ax = plt.subplots(figsize=(3.5, 2.6))

ax.axvspan(etas_pct[0] * 0.6, eta_max_pct, color=ZONE, alpha=0.07, zorder=0)
ax.axvline(eta_max_pct, color=RED, linestyle="--", linewidth=1.0, zorder=2)
ax.text(eta_max_pct * 1.12, 38, r"guaranteed regime (Thm. 5)"
        "\n" r"$\eta^\star=\mu/M\approx%.1f\%%$" % eta_max_pct,
        color=RED, fontsize=7, va="center", ha="left")

ax.errorbar(etas_pct, taux, yerr=[bas, haut], fmt="o-", color=BLUE,
            markersize=3.5, linewidth=1.0, markeredgecolor="white",
            markeredgewidth=0.4, capsize=2, elinewidth=0.8, zorder=3)
ax.text(etas_pct[0], 14, "empirical exact\nrecovery", color=BLUE,
        fontsize=7, va="center", ha="left")

ax.set_xscale("log")
ticks_pct = [1, 2, 5, 10, 20, 30]
ax.xaxis.set_major_locator(FixedLocator(ticks_pct))
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.xaxis.set_minor_locator(FixedLocator([]))
ax.xaxis.set_minor_formatter(NullFormatter())

ax.set_xlabel(r"noise level $\eta$ (%)")
ax.set_ylabel("exact recovery rate (%)")
ax.set_xlim(etas_pct[0] * 0.75, etas_pct[-1] * 1.25)
ax.set_ylim(-2, 104)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.grid(True, which="major", axis="y", alpha=0.2, linewidth=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout(pad=0.4)
fig.savefig("figures/succes_vs_eta_reel_n4.pdf", bbox_inches="tight")
fig.savefig("figures/succes_vs_eta_reel_n4.png", dpi=300, bbox_inches="tight")
print("-> figures/succes_vs_eta_reel_n4.pdf")
