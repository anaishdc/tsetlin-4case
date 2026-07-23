""" Figure Exp 2 (validation du Theoreme 3, robustesse au bruit) : axe X
= bruit reel eta, ligne verticale = valeur exacte de mu/M, pour UNE
cible concrete fixee (n=8, a=1.0 b=0.3 c=1.0 d=0.1 -- nos parametres
habituels). Plus concrete/lisible que le graphe agrege sur plusieurs n
(r=eta/eta_max) : montre directement les vrais nombres pour un exemple
precis.
"""

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 11,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
})
import matplotlib.pyplot as plt

eta_max = 0.00481
etas = [0.00481, 0.00962, 0.02404, 0.04808, 0.09615, 0.14423, 0.19231,
        0.24038, 0.28846, 0.33654, 0.38462, 0.43269, 0.48077]
taux = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
        10.0, 0.0, 0.0, 0.0, 0.0]

fig, ax = plt.subplots(figsize=(3.6, 2.9))
ax.plot(etas, taux, marker="o", markersize=4, linewidth=1.4,
        color="#1f5fa8", zorder=3)
ax.axvline(eta_max, color="#c0392b", linestyle="--", linewidth=1.1, zorder=2)
ax.text(eta_max * 1.3, 6, r"$\eta = \mu/M$", color="#c0392b", fontsize=9, va="bottom")

ax.set_xscale("log")
ax.set_xlabel(r"noise level $\eta$")
ax.set_ylabel("exact recovery rate (%)")
ax.set_ylim(-4, 104)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.grid(True, which="major", alpha=0.25, linewidth=0.6)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout(pad=0.4)
fig.savefig("figures/succes_vs_eta_reel_n8.pdf")
fig.savefig("figures/succes_vs_eta_reel_n8.png", dpi=300)
print("-> figures/succes_vs_eta_reel_n8.pdf")
