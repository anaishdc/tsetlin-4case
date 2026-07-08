import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11})

# ---------------------------------------------------------------
# Test A : seuil sur S
# ---------------------------------------------------------------
dfA = pd.read_csv("test_A_seuil_S.csv")
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(dfA["S"], dfA["piV_Inc_theorie"], "-", color="tab:blue", label=r"$\pi_v(\mathrm{Include})$ théorique")
ax.plot(dfA["S"], dfA["vIncludeRate_empirique"], "o", color="tab:red", label="empirique (20 runs)")
ax.axvline(0.75, color="gray", linestyle="--", linewidth=1.5, label=r"$S^*=0{,}75$")
ax.set_xlabel("S")
ax.set_ylabel("taux d'inclusion de $v$")
ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.4)
ax.legend(loc="upper left")
ax.set_title("Test A : seuil sur S (noiseProb=0.3, alpha=1, N=50)")
fig.tight_layout()
fig.savefig("figures/test_A_seuil_S.png", dpi=150)
fig.savefig("figures/test_A_seuil_S.pdf")
plt.close(fig)

# ---------------------------------------------------------------
# Test B : limite de bruit tolerable
# ---------------------------------------------------------------
dfB = pd.read_csv("test_B_effet_bruit.csv")
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(dfB["noiseProb"], dfB["piV_Inc_theorie"], "-", color="tab:blue", label=r"$\pi_v(\mathrm{Include})$ théorique")
ax.plot(dfB["noiseProb"], dfB["vIncludeRate_empirique"], "o", color="tab:red", label="empirique (20 runs)")
ax.axvline(1/3, color="gray", linestyle="--", linewidth=1.5, label=r"$1/3$")
ax.set_xlabel("bruit (noiseProb)")
ax.set_ylabel("taux d'inclusion de $v$")
ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.4)
ax.legend(loc="center left")
ax.set_title("Test B : limite de bruit tolérable (S=1, alpha=1, N=50)")
fig.tight_layout()
fig.savefig("figures/test_B_effet_bruit.png", dpi=150)
fig.savefig("figures/test_B_effet_bruit.pdf")
plt.close(fig)

# ---------------------------------------------------------------
# Test C : role de N
# ---------------------------------------------------------------
dfC = pd.read_csv("test_C_effet_N.csv")
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(dfC["N"], dfC["erreur_theorie"], "-", color="tab:blue", label="erreur théorique")
ax.plot(dfC["N"], dfC["erreur_empirique"], "o", color="tab:red", label="erreur empirique (20 runs)")
ax.set_yscale("log")
ax.set_xlabel("N")
ax.set_ylabel(r"erreur $1-\pi_v(\mathrm{Include})$")
ax.grid(True, which="both", alpha=0.4)
ax.legend(loc="upper right")
ax.set_title("Test C : rôle de N (S=1, noiseProb=0.3, alpha=1)")
fig.tight_layout()
fig.savefig("figures/test_C_effet_N.png", dpi=150)
fig.savefig("figures/test_C_effet_N.pdf")
plt.close(fig)

print("OK -> figures/test_A_seuil_S.{png,pdf}, test_B_effet_bruit.{png,pdf}, test_C_effet_N.{png,pdf}")
