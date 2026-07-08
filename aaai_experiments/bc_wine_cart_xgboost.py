#!/usr/bin/env python3
# CART + XGBoost sur Breast Cancer (protocole apparie a 20 runs, reshuffle 80/20 comme Iris)
import json
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

PARAM_GRID = {"n_estimators": [16, 50, 100, 300], "max_depth": [1, 2, 3, 4], "learning_rate": [0.1, 0.3]}

def load_reshuffled(path, n_features, seed):
    data = np.loadtxt(path).astype(int)
    rng = np.random.RandomState(20000 + seed)
    d = data.copy(); rng.shuffle(d)
    split = int(round(0.8 * len(d)))
    return d[:split, :n_features], d[:split, n_features], d[split:, :n_features], d[split:, n_features]

def cart_sweep(path, n_features, Ks, n_runs=20):
    out = {}
    for K in Ks:
        accs = []
        for seed in range(n_runs):
            Xtr, ytr, Xte, yte = load_reshuffled(path, n_features, seed)
            clf = DecisionTreeClassifier(max_leaf_nodes=max(2, K), random_state=seed)
            clf.fit(Xtr, ytr)
            accs.append(100.0 * clf.score(Xte, yte))
        out[K] = (float(np.mean(accs)), float(np.std(accs)))
    return out

def xgboost_matched(path, n_features, n_runs=20):
    Xtr0, ytr0, _, _ = load_reshuffled(path, n_features, 0)
    base = XGBClassifier(objective="binary:logistic", eval_metric="logloss", n_jobs=4, verbosity=0)
    gs = GridSearchCV(base, PARAM_GRID, cv=3, n_jobs=4, scoring="accuracy")
    gs.fit(Xtr0, ytr0)
    params = gs.best_params_
    accs = []
    for seed in range(n_runs):
        Xtr, ytr, Xte, yte = load_reshuffled(path, n_features, seed)
        clf = XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=seed, **params)
        clf.fit(Xtr, ytr)
        accs.append(100.0 * clf.score(Xte, yte))
    return float(np.mean(accs)), float(np.std(accs)), params

if __name__ == "__main__":
    results = {"cart": {}, "xgboost": {}}
    print("=== Breast Cancer : CART ===")
    cart = cart_sweep("../data/BinaryBreastCancerData.txt", 120, [2, 4, 8])
    for K, (m, s) in cart.items(): print(f"  K={K}: {m:.1f} +/- {s:.1f}")
    results["cart"]["breastcancer"] = cart

    print("=== Breast Cancer : XGBoost (20 runs, protocole apparie) ===")
    m, s, params = xgboost_matched("../data/BinaryBreastCancerData.txt", 120, 20)
    print(f"  -> {m:.1f} +/- {s:.1f}  {params}")
    results["xgboost"]["breastcancer"] = m

    with open("bc_results.json", "w") as f:
        json.dump({"cart": {k: {str(kk): vv for kk, vv in v.items()} for k, v in results["cart"].items()},
                    "xgboost": results["xgboost"]}, f, indent=2)
    print("Ecrit : bc_results.json")
