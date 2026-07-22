import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11})

# ---------------------------------------------------------------
# Courbe theorique : seuil alpha*(bruit), a S=1 fixe
# ---------------------------------------------------------------
dfSeuil = pd.read_csv("test_D_alpha_seuils.csv")
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(dfSeuil["noiseProb"], dfSeuil["alphaStar_theorie"], "-",
        color="tab:blue", label=r"$\alpha^*(b)$ théorique")
ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.2,
           label=r"$\alpha=1$ (plafond physique)")
ax.set_xlabel("bruit $b$ = noiseProb")
ax.set_ylabel(r"seuil $\alpha^*$ ($S=1$ fixe)")
ax.set_ylim(0, 5)
ax.grid(True, alpha=0.4)
ax.legend(loc="upper right")
ax.set_title(r"Seuil théorique $\alpha^*$ en fonction du bruit (S=1)")
fig.tight_layout()
fig.savefig("figures/test_D_alpha_seuils.png", dpi=150)
fig.savefig("figures/test_D_alpha_seuils.pdf")
plt.close(fig)

# ---------------------------------------------------------------
# Balayage empirique de alpha, pour chaque niveau de bruit fixe
# ---------------------------------------------------------------
dfBal = pd.read_csv("test_D_alpha_par_bruit.csv")
noises = sorted(dfBal["noiseProb"].unique())

fig, axes = plt.subplots(1, len(noises), figsize=(3.6 * len(noises), 4.2), sharey=True)
for ax, noise in zip(axes, noises):
    sub = dfBal[dfBal["noiseProb"] == noise]
    alphaStar = sub["alphaStar_theorie"].iloc[0]
    ax.plot(sub["alpha"], sub["piV_Inc_theorie"], "-", color="tab:blue",
            label=r"$\pi_v(\mathrm{Include})$ théorique")
    ax.plot(sub["alpha"], sub["vIncludeRate_empirique"], "o", color="tab:red",
            label="empirique (20 runs)")
    if 0 < alphaStar <= 1.05:
        ax.axvline(alphaStar, color="gray", linestyle="--", linewidth=1.2, label=r"$\alpha^*$")
    ax.set_xlabel(r"$\alpha$")
    title_star = rf"$\alpha^*$={alphaStar:.3f}" if alphaStar > 0 else r"$\alpha^*$ n/a"
    ax.set_title(f"bruit={noise:.2f}\n{title_star}")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.4)

axes[0].set_ylabel("taux d'inclusion de $v$")
axes[0].legend(loc="center left", fontsize=9)
fig.suptitle(r"Test D : balayage de $\alpha$ pour différents niveaux de bruit ($S=1$, $N=50$)")
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig("figures/test_D_alpha_par_bruit.png", dpi=150)
fig.savefig("figures/test_D_alpha_par_bruit.pdf")
plt.close(fig)

print("OK -> figures/test_D_alpha_seuils.{png,pdf}, test_D_alpha_par_bruit.{png,pdf}")
