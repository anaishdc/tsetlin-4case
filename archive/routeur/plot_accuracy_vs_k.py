#!/usr/bin/env python3
# -----------------------------------------------------------------
# Graphe accuracy vs K : notre systeme complet, notre baseline
# (routeur seul), CART (arbre isole) et XGBoost -- Iris, XOR, Digits.
# Version "professionnelle" (legende propre, typographie soignee).
# -----------------------------------------------------------------
import json
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

INK       = "#171b18"
MUTED     = "#52514e"
GRID      = "#d8d7d1"
COL_SYSTEME  = "#2a78d6"
COL_BASELINE = "#1baf7a"
COL_CART     = "#eda100"
COL_XGB      = "#4a3aa7"
COL_VRAITM   = "#e34948"

OURS = {
    "iris":   {2: (93.1, 4.5, 52.7, 23.9), 4: (94.0, 3.5, 92.0, 4.7), 8: (95.1, 4.4, 95.3, 3.8)},
    "xor":    {2: (100.0, 0.0, 50.1, 0.0), 4: (85.4, 7.2, 44.8, 12.0), 8: (69.1, 8.2, 46.8, 20.5)},
    "digits": {2: (92.4, 0.1, 9.2, 0.0), 4: (96.5, 0.7, 59.7, 3.3), 8: (97.2, 0.3, 91.7, 0.6),
               16: (97.8, 0.3, 96.5, 0.1), 32: (96.5, 0.4, 96.4, 0.6)},
    "mnist":  {16: (84.5, None, 38.6, None), 32: (89.3, None, 43.0, None), 64: (92.3, None, 56.3, None)},
}
XGBOOST = {"iris": 91.8, "xor": 60.9, "digits": 97.2, "mnist": 98.2}
# Note : la vraie Tsetlin Machine officielle n'a pas de parametre K (son
# hyperparametre equivalent est le nombre de clauses) -- pas de sens de la
# tracer sur ce graphe. Voir accuracy_vs_complexity.png pour cette comparaison.

with open("cart_results.json") as f:
    cart = json.load(f)
with open("mnist_cart.json") as f:
    cart["mnist"] = json.load(f)

datasets = [("iris", "Iris"), ("xor", "Noisy XOR"), ("digits", "Digits (8×8)"), ("mnist", "MNIST")]

fig, axes = plt.subplots(1, 4, figsize=(17.5, 5.2))
fig.patch.set_facecolor("white")

for ax, (key, title) in zip(axes, datasets):
    ax.set_facecolor("white")
    Ks_cart = sorted(int(k) for k in cart[key].keys())
    cart_means = [cart[key][str(k)][0] if isinstance(cart[key][str(k)], list) else cart[key][str(k)]
                  for k in Ks_cart]
    ax.plot(Ks_cart, cart_means, color=COL_CART, lw=2, marker="o",
            markersize=6.5, markeredgewidth=1.4, markeredgecolor="white", zorder=3)

    Ks = sorted(OURS[key].keys())
    sys_m = [OURS[key][k][0] for k in Ks]
    sys_s = [OURS[key][k][1] if OURS[key][k][1] is not None else 0 for k in Ks]
    ax.plot(Ks, sys_m, color=COL_SYSTEME, lw=2.6, marker="o",
             markersize=7.5, markeredgewidth=1.4, markeredgecolor="white", zorder=4)
    ax.fill_between(Ks, np.array(sys_m) - np.array(sys_s), np.array(sys_m) + np.array(sys_s),
                     color=COL_SYSTEME, alpha=0.12, linewidth=0, zorder=1)

    base_Ks = [k for k in Ks if OURS[key][k][2] is not None]
    if len(base_Ks) >= 2:
        base_m = [OURS[key][k][2] for k in base_Ks]
        base_s = [OURS[key][k][3] or 0 for k in base_Ks]
        ax.plot(base_Ks, base_m, color=COL_BASELINE, lw=2.2, marker="o", linestyle=(0, (5, 2)),
                 markersize=6.5, markeredgewidth=1.4, markeredgecolor="white", zorder=3)
        ax.fill_between(base_Ks, np.array(base_m) - np.array(base_s), np.array(base_m) + np.array(base_s),
                         color=COL_BASELINE, alpha=0.10, linewidth=0, zorder=1)
    elif len(base_Ks) == 1:
        ax.plot(base_Ks, [OURS[key][base_Ks[0]][2]], color=COL_BASELINE, marker="o",
                 markersize=7, markeredgewidth=1.4, markeredgecolor="white", zorder=3)

    if key in XGBOOST:
        ax.axhline(XGBOOST[key], color=COL_XGB, lw=1.8, linestyle=(0, (1, 1.6)), zorder=2)

    ax.set_title(title, fontsize=13.5, fontweight="bold", color=INK, pad=10)
    ax.set_xlabel("K  (nombre de feuilles)", fontsize=10.5, color=MUTED)
    if ax is axes[0]:
        ax.set_ylabel("Accuracy — ensemble de test (%)", fontsize=10.5, color=MUTED)
    ax.set_ylim(0, 104)
    ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9.5, length=0)

fig.suptitle("Accuracy en fonction du nombre de feuilles (K)",
             fontsize=16, fontweight="bold", color=INK, y=0.985, x=0.02, ha="left")
fig.text(0.02, 0.925,
         "Notre méthode (routeur + clauses + AdaBoost) comparée à un arbre isolé (CART) et à XGBoost, à budget de feuilles égal.",
         fontsize=10.5, color=MUTED, ha="left")

legend_handles = [
    mlines.Line2D([], [], color=COL_SYSTEME, lw=2.6, marker="o", markersize=7,
                  markeredgewidth=1.2, markeredgecolor="white", label="Notre système (routeur + clauses + boosting)"),
    mlines.Line2D([], [], color=COL_BASELINE, lw=2.2, linestyle=(0, (5, 2)), marker="o", markersize=6,
                  markeredgewidth=1.2, markeredgecolor="white", label="Notre baseline (routeur seul + boosting, sans clause)"),
    mlines.Line2D([], [], color=COL_CART, lw=2, marker="o", markersize=6,
                  markeredgewidth=1.2, markeredgecolor="white", label="CART (arbre isolé, sklearn, sans boosting)"),
    mlines.Line2D([], [], color=COL_XGB, lw=1.8, linestyle=(0, (1, 1.6)), label="XGBoost (meilleure configuration trouvée)"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=2, frameon=False,
           fontsize=10, bbox_to_anchor=(0.5, -0.09), labelcolor=INK,
           handlelength=2.6, columnspacing=1.8, handletextpad=0.7)

fig.tight_layout(rect=[0, 0.06, 1, 0.87])
fig.savefig("accuracy_vs_k.png", dpi=200, bbox_inches="tight", facecolor="white")
print("Ecrit : accuracy_vs_k.png")
