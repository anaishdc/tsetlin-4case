""" Genere la figure de l'Experience 2 (robustesse au bruit) au format
publication, prete a inclure dans le papier AAAI (PDF vectoriel, police
Computer Modern, sans titre integre -- la legende va dans \caption{}).

Reutilise les resultats deja calcules par graphe_bruit.py (200 essais
par valeur de r, N=3000, a=1.0 b=0.3 c=1.0 d=0.1).
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

rs = [0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
taux_succes = [100.0, 100.0, 100.0, 99.5, 98.0, 95.0, 87.0, 68.0, 23.5, 0.0, 0.0]

fig, ax = plt.subplots(figsize=(3.6, 2.9))

ax.plot(rs, taux_succes, marker="o", markersize=4, linewidth=1.4,
        color="#1f5fa8", zorder=3)
ax.axvline(1.0, color="#c0392b", linestyle="--", linewidth=1.1, zorder=2)
ax.text(1.08, 6, r"$r=1$", color="#c0392b", fontsize=10, va="bottom")

ax.set_xscale("log")
ax.set_xlabel(r"$r = \eta / \eta_{\max}$")
ax.set_ylabel("exact recovery rate (%)")
ax.set_xlim(0.15, 700)
ax.set_ylim(-4, 104)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.grid(True, which="major", alpha=0.25, linewidth=0.6)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout(pad=0.4)
fig.savefig("figures/succes_vs_r_theoreme5_papier.pdf")
fig.savefig("figures/succes_vs_r_theoreme5_papier.png", dpi=300)
print("-> figures/succes_vs_r_theoreme5_papier.pdf (et .png pour previsualiser)")
