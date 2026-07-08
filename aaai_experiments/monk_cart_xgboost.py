#!/usr/bin/env python3
# -----------------------------------------------------------------
# CART + XGBoost sur Monk-1/2/3, splits FIXES officiels (comme notre
# systeme), meme protocole apparie que pour Iris/XOR/Digits :
#   - CART : max_leaf_nodes=K, direct sur le split fixe
#   - XGBoost : 1 recherche d'hyperparametres (CV sur train), puis
#     figee et evaluee sur nbRuns=20 seeds d'entrainement (meme
#     nombre que notre systeme sur Monk).
# -----------------------------------------------------------------
import json
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

PARAM_GRID = {
    "n_estimators": [16, 50, 100, 300],
    "max_depth": [1, 2, 3, 4],
    "learning_rate": [0.1, 0.3],
}

def load(monk):
    tr = np.loadtxt(f"../data/Monk{monk}TrainingData.txt").astype(int)
    te = np.loadtxt(f"../data/Monk{monk}TestData.txt").astype(int)
    n = 17
    return tr[:, :n], tr[:, n], te[:, :n], te[:, n]

def cart_sweep(monk, Ks):
    Xtr, ytr, Xte, yte = load(monk)
    out = {}
    for K in Ks:
        clf = DecisionTreeClassifier(max_leaf_nodes=max(2, K), random_state=0)
        clf.fit(Xtr, ytr)
        out[K] = 100.0 * clf.score(Xte, yte)
    return out

def xgboost_matched(monk, n_runs=20):
    Xtr, ytr, Xte, yte = load(monk)
    base = XGBClassifier(objective="binary:logistic", eval_metric="logloss", n_jobs=4, verbosity=0)
    gs = GridSearchCV(base, PARAM_GRID, cv=3, n_jobs=4, scoring="accuracy")
    gs.fit(Xtr, ytr)
    params = gs.best_params_
    accs = []
    for seed in range(n_runs):
        clf = XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                             random_state=seed, **params)
        clf.fit(Xtr, ytr)
        accs.append(100.0 * clf.score(Xte, yte))
    return float(np.mean(accs)), float(np.std(accs)), params

if __name__ == "__main__":
    results = {"cart": {}, "xgboost": {}}
    for monk in [1, 2, 3]:
        key = f"monk{monk}"
        print(f"=== Monk-{monk} : CART ===")
        cart = cart_sweep(monk, [2, 4, 8])
        for K, acc in cart.items(): print(f"  K={K}: {acc:.1f}%")
        results["cart"][key] = cart

        print(f"=== Monk-{monk} : XGBoost (20 runs, protocole apparie) ===")
        m, s, params = xgboost_matched(monk, 20)
        print(f"  -> {m:.1f} +/- {s:.1f}  {params}")
        results["xgboost"][key] = m

    with open("monk_results.json", "w") as f:
        json.dump({
            "cart": {k: {str(kk): vv for kk, vv in v.items()} for k, v in results["cart"].items()},
            "xgboost": results["xgboost"],
        }, f, indent=2)
    print("\nEcrit : monk_results.json")
