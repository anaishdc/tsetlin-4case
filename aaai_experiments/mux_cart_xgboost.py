#!/usr/bin/env python3
# CART/XGBoost sur Mux6/Mux11 (meme protocole matched-runs que XOR/Iris/Digits).
import json
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

def load(train_path, test_path, n):
    tr = np.loadtxt(train_path).astype(int)
    te = np.loadtxt(test_path).astype(int)
    return tr[:, :n], tr[:, n], te[:, :n], te[:, n]

results = {"cart": {}, "xgboost": {}}

for name, n in [("mux6", 6), ("mux11", 11)]:
    Xtr, ytr, Xte, yte = load(f"../data/Mux{n}TrainingData.txt", f"../data/Mux{n}TestData.txt", n)

    results["cart"][name] = {}
    for K in [2, 4, 8, 16, 32, 64]:
        clf = DecisionTreeClassifier(max_leaf_nodes=K, random_state=0).fit(Xtr, ytr)
        acc = 100.0 * clf.score(Xte, yte)
        results["cart"][name][K] = acc
    print(name, "CART:", results["cart"][name], flush=True)

    grid = {"n_estimators": [50, 100, 300], "max_depth": [2, 3, 4], "learning_rate": [0.1, 0.3]}
    base = XGBClassifier(objective="binary:logistic", eval_metric="logloss", n_jobs=4, verbosity=0)
    gs = GridSearchCV(base, grid, cv=3, n_jobs=-1, scoring="accuracy")
    gs.fit(Xtr, ytr)
    acc = 100.0 * gs.best_estimator_.score(Xte, yte)
    results["xgboost"][name] = {"acc": acc, "params": gs.best_params_}
    print(name, "XGBoost:", results["xgboost"][name], flush=True)

json.dump(results, open("mux_results.json", "w"), indent=2)
print("Ecrit : mux_results.json")
