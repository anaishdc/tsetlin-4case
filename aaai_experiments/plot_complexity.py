#!/usr/bin/env python3
# -----------------------------------------------------------------
# Graphe accuracy vs complexite (log) : notre systeme (point unique,
# a son K valide) vs CART (plusieurs elagages) vs XGBoost (plusieurs
# n_estimators) -- Iris, XOR, Digits.
# Complexite = nb total de conditions booleennes actives dans le
# modele entier (noeuds du routeur + litteraux Include pour nous ;
# noeuds internes pour CART ; noeuds internes somme sur tous les
# arbres pour XGBoost).
# -----------------------------------------------------------------
import json
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

INK   = "#171b18"
MUTED = "#52514e"
GRID  = "#d8d7d1"
COL_SYSTEME  = "#2a78d6"
COL_BASELINE = "#1baf7a"
COL_CART     = "#eda100"
COL_XGB      = "#4a3aa7"
COL_VRAITM   = "#e34948"

# Notre systeme : point unique par dataset, a la config deja validee
# (meme K/M que dans accuracy_vs_k.png). complexite totale = noeuds
# routeur + litteraux Include, somme sur tout l'ensemble (M rounds x classes).
OURS = {
    "iris":   {"acc": 93.1,  "total_complexity": 48 + 434,     "decision_complexity": 300.6},
    "xor":    {"acc": 100.0, "total_complexity": 5 + 2,        "decision_complexity": 6.0},
    "digits": {"acc": 96.5,  "total_complexity": 480 + 40273,  "decision_complexity": 11819.2},
    "mux11":  {"acc": 82.7,  "total_complexity": 35 + 26,      "decision_complexity": 17.9},
}
# Notre baseline (routeur seul, sans clause) -- meme M/K que OURS,
# complexite totale = noeuds routeur uniquement (pas de litteraux).
BASELINE = {
    "iris":   {"acc": 52.7, "total_complexity": 48},
    "xor":    {"acc": 50.1, "total_complexity": 5},
    "digits": {"acc": 59.7, "total_complexity": 480},
    "mux11":  {"acc": 57.8, "total_complexity": 35},
}

with open("complexity_results.json") as f:
    comp = json.load(f)
with open("realtm_complexity.json") as f:
    realtm = json.load(f)
with open("complexity_mux11.json") as f:
    mux = json.load(f)
    comp["cart"]["mux11"] = mux["cart"]["mux11"]
    comp["xgboost"]["mux11"] = mux["xgboost"]["mux11"]

datasets = [("iris", "Iris"), ("xor", "Noisy XOR"), ("digits", "Digits (8×8)"), ("mux11", "Multiplexeur 11-bit")]

fig, axes = plt.subplots(1, 4, figsize=(18, 5.2))
fig.patch.set_facecolor("white")

for ax, (key, title) in zip(axes, datasets):
    ax.set_facecolor("white")

    cart_pts = sorted(comp["cart"][key], key=lambda r: r["total_complexity"])
    xgb_pts  = sorted(comp["xgboost"][key], key=lambda r: r["total_complexity"])

    ax.plot([p["total_complexity"] for p in cart_pts], [p["acc"] for p in cart_pts],
            color=COL_CART, lw=2, marker="o", markersize=6.5,
            markeredgewidth=1.4, markeredgecolor="white", zorder=3)
    ax.plot([p["total_complexity"] for p in xgb_pts], [p["acc"] for p in xgb_pts],
            color=COL_XGB, lw=2, marker="o", markersize=6.5,
            markeredgewidth=1.4, markeredgecolor="white", zorder=3, linestyle=(0, (1, 1.6)))

    ax.plot(OURS[key]["total_complexity"], OURS[key]["acc"], color=COL_SYSTEME,
            marker="*", markersize=20, markeredgewidth=1.2, markeredgecolor="white", zorder=5,
            linestyle="none")
    ax.plot(BASELINE[key]["total_complexity"], BASELINE[key]["acc"], color=COL_BASELINE,
            marker="D", markersize=9, markeredgewidth=1.2, markeredgecolor="white", zorder=5,
            linestyle="none")
    if key in realtm:
        ax.plot(realtm[key]["total_complexity"], realtm[key]["acc"], color=COL_VRAITM,
                marker="P", markersize=12, markeredgewidth=1.2, markeredgecolor="white", zorder=5,
                linestyle="none")

    ax.set_xscale("log")
    ax.set_title(title, fontsize=13.5, fontweight="bold", color=INK, pad=10)
    ax.set_xlabel("Complexité totale du modèle (log)", fontsize=10.5, color=MUTED)
    if ax is axes[0]:
        ax.set_ylabel("Accuracy — ensemble de test (%)", fontsize=10.5, color=MUTED)
    ax.set_ylim(0, 104)
    ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
    ax.grid(axis="x", color=GRID, linewidth=0.6, zorder=0, which="both", alpha=0.4)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9.5, length=0)

fig.suptitle("Accuracy en fonction de la complexité du modèle",
             fontsize=16, fontweight="bold", color=INK, y=0.985, x=0.02, ha="left")
fig.text(0.02, 0.925,
         "Complexité = nombre total de conditions booléennes actives dans tout le modèle "
         "(nœuds de routage + littéraux, ou nœuds d'arbre, ou littéraux Include de la vraie TM). "
         "Notre système, notre baseline et la vraie TM : un seul point chacun, à leur configuration déjà validée.",
         fontsize=10.5, color=MUTED, ha="left")

legend_handles = [
    mlines.Line2D([], [], color=COL_SYSTEME, marker="*", markersize=14, linestyle="none",
                  markeredgewidth=1, markeredgecolor="white", label="Notre système (point unique, K validé)"),
    mlines.Line2D([], [], color=COL_BASELINE, marker="D", markersize=8, linestyle="none",
                  markeredgewidth=1, markeredgecolor="white", label="Notre baseline (routeur seul, sans clause)"),
    mlines.Line2D([], [], color=COL_VRAITM, marker="P", markersize=10, linestyle="none",
                  markeredgewidth=1, markeredgecolor="white", label="Vraie Tsetlin Machine officielle"),
    mlines.Line2D([], [], color=COL_CART, lw=2, marker="o", markersize=6,
                  markeredgewidth=1.2, markeredgecolor="white", label="CART (plusieurs niveaux d'élagage)"),
    mlines.Line2D([], [], color=COL_XGB, lw=2, linestyle=(0, (1, 1.6)), marker="o", markersize=6,
                  markeredgewidth=1.2, markeredgecolor="white", label="XGBoost (plusieurs n_estimators)"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False,
           fontsize=10, bbox_to_anchor=(0.5, -0.08), labelcolor=INK,
           handlelength=2.6, columnspacing=1.8, handletextpad=0.7)

fig.tight_layout(rect=[0, 0.1, 1, 0.87])
fig.savefig("accuracy_vs_complexity.png", dpi=200, bbox_inches="tight", facecolor="white")
print("Ecrit : accuracy_vs_complexity.png")
