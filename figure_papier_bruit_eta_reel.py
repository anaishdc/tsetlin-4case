""" Figure Exp 2 (validation du Theoreme 3, robustesse au bruit) : axe X
= bruit reel eta, ligne verticale = valeur exacte de mu/M, pour UNE
cible concrete fixee (n=4, cible=[1,2,3,4], a=1.0 b=0.3 c=1.0 d=0.1 --
nos parametres habituels). Version "pro" : zone garantie ombree, legende
propre, echelle log avec ticks lisibles.
"""

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 12,
    "axes.linewidth": 0.9,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": False,
    "legend.frameon": False,
})
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

eta_max = 0.05288
etas = [0.01058, 0.02644, 0.05288, 0.10577, 0.15865, 0.21154, 0.26442, 0.31731]
taux = [100.0, 100.0, 100.0, 100.0, 100.0, 60.0, 20.0, 0.0]

BLUE = "#1f5fa8"
RED = "#c0392b"
GREEN_FILL = "#2e7d32"

fig, ax = plt.subplots(figsize=(4.2, 3.2))

# zone garantie par le Theoreme 3 (eta < mu/M)
ax.axvspan(etas[0] * 0.5, eta_max, color=GREEN_FILL, alpha=0.08, zorder=0)
ax.axvline(eta_max, color=RED, linestyle="--", linewidth=1.3, zorder=2,
           label=r"threshold $\eta = \mu/M$")

ax.plot(etas, taux, marker="o", markersize=5.5, linewidth=1.6,
        color=BLUE, markeredgecolor="white", markeredgewidth=0.6,
        zorder=3, label="exact recovery rate")

ax.set_xscale("log")
ax.xaxis.set_major_locator(LogLocator(base=10))
ax.xaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))
ax.xaxis.set_minor_formatter(NullFormatter())

ax.set_xlabel(r"noise level $\eta$")
ax.set_ylabel("exact recovery rate (%)")
ax.set_xlim(etas[0] * 0.7, etas[-1] * 1.5)
ax.set_ylim(-4, 104)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.grid(True, which="major", axis="y", alpha=0.25, linewidth=0.6)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(loc="lower left", fontsize=9.5, handlelength=1.6, borderaxespad=0.3)

fig.tight_layout(pad=0.4)
fig.savefig("figures/succes_vs_eta_reel_n4.pdf")
fig.savefig("figures/succes_vs_eta_reel_n4.png", dpi=300)
print("-> figures/succes_vs_eta_reel_n4.pdf")
