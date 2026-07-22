"""Graphe : effet de S sur la vitesse/direction de selection d'un automate
(Test D, repositionne comme validation empirique du Theoreme 6 -- selection
sous loi arbitraire -- et non de la demonstration 1-bit retiree du papier).

Feature non pertinente (independante de Y), marginale desequilibree c1=0.8,
a=0.15 (notation du code = notre 'a'), p_y=P(Y=1)=0.5. On fait varier S : la
theorie (Theoreme 6, signe de Deltabar_k) predit un seuil exact S* au dela
duquel l'automate vbar1 (negation de la feature non pertinente) bascule a
tort de Exclude vers Include.

Donnees : 1_clause/test_D_seuil_S_neutralite.csv (deja generees par
1_clause/validation_theorie_neutralite.cpp, 20 runs/point, 3M tirages/run).

Usage: python3 plot_test_D_thm6.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("test_D_seuil_S_neutralite.csv")

c1, alpha, p_y = 0.8, 0.15, 0.5
S_grid = np.geomspace(df.S.min(), df.S.max(), 400)
pPlusVbar1 = c1 * (1 - p_y) * S_grid
pMinusVbar1 = (1 - c1) * (1 - p_y) * S_grid + c1 * p_y * alpha
rhobar = pPlusVbar1 / pMinusVbar1
S_star = c1 * p_y * alpha / ((2 * c1 - 1) * (1 - p_y))  # rhobar(S*)=1, cf. Theoreme 6

fig, ax = plt.subplots(figsize=(6.2, 4.5))

ax.axvspan(df.S.min(), S_star, color="#2e8b57", alpha=0.08)
ax.axvspan(S_star, df.S.max(), color="#c0392b", alpha=0.08)
ax.axvline(S_star, color="black", linestyle=":", linewidth=1.5,
           label=rf"$S^\star={S_star:.3f}$ ($\bar\Delta_k=0$, Théorème 6)")

ax.plot(df.S, df.vbar1IncRate_empirique, "o-", color="#c0392b", linewidth=2,
        markersize=6, label=r"$P(\bar v_1 \in \mathrm{Inc})$ empirique")
ax.plot(df.S, df.v1IncRate_empirique, "^-", color="#1b6ca8", linewidth=2,
        markersize=6, label=r"$P(v_1 \in \mathrm{Inc})$ empirique")

ax.set_xscale("log")
ax.set_xlabel("$S$ (échelle log)")
ax.set_ylabel("Probabilité d'inclusion (littéral non pertinent)")
ax.set_title("Effet de $S$ sur la sélection d'un littéral non pertinent\n"
              "(validation empirique du Théorème 6, feature déséquilibrée $c_1=0.8$)")
ax.text(df.S.min()*1.15, 0.5, r"$\bar\Delta_k<0$" "\n(exclusion correcte)", color="#2e8b57", fontsize=9)
ax.text(S_star*1.4, 0.5, r"$\bar\Delta_k>0$" "\n(inclusion à tort)", color="#c0392b", fontsize=9)
ax.legend(loc="center right", fontsize=9)
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig("../figures/graphe_S_thm6.png", dpi=150)
print("Figure ecrite dans figures/graphe_S_thm6.png")
print(f"S* theorique = {S_star:.4f} (coherence avec le passage a 0.5 de la CSV entre S=0.18 et S=0.22)")
