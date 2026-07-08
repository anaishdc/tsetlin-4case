#!/usr/bin/env python3
# Complexite CART/XGBoost sur Mux11 (meme protocole que complexity_cart_xgboost.py).
import json
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

def cart_complexity(clf, X):
    n_internal = clf.tree_.node_count - clf.get_n_leaves()
    paths = clf.decision_path(X)
    avg_depth = paths.sum(axis=1).mean() - 1
    return n_internal, float(avg_depth)

def xgb_complexity(clf, n_estimators):
    df = clf.get_booster().trees_to_dataframe()
    n_internal = int((df["Feature"] != "Leaf").sum())
    depths = []
    for tid in df["Tree"].unique():
        sub = df[df["Tree"] == tid]
        depths.append(sub["Feature"].ne("Leaf").sum())
    avg_depth_per_tree = np.mean(depths) if depths else 0
    return n_internal, n_estimators * avg_depth_per_tree

n = 11
tr = np.loadtxt("../data/Mux11TrainingData.txt").astype(int)
te = np.loadtxt("../data/Mux11TestData.txt").astype(int)
Xtr, ytr = tr[:, :n], tr[:, n]
Xte, yte = te[:, :n], te[:, n]

results = {"cart": {"mux11": []}, "xgboost": {"mux11": []}}

for K in [2, 4, 8, 16, 32, 64]:
    clf = DecisionTreeClassifier(max_leaf_nodes=K, random_state=0).fit(Xtr, ytr)
    n_int, depth = cart_complexity(clf, Xte)
    acc = 100.0 * clf.score(Xte, yte)
    results["cart"]["mux11"].append({"K": K, "acc": acc, "total_complexity": n_int, "decision_complexity": depth})
print("Mux11 CART:", results["cart"]["mux11"])

for ne in [1, 4, 16, 50, 100]:
    clf = XGBClassifier(objective="binary:logistic", n_estimators=ne, max_depth=4, learning_rate=0.1,
                         eval_metric="logloss", n_jobs=4, verbosity=0).fit(Xtr, ytr)
    n_int, dc = xgb_complexity(clf, ne)
    acc = 100.0 * clf.score(Xte, yte)
    results["xgboost"]["mux11"].append({"n_estimators": ne, "acc": acc, "total_complexity": n_int, "decision_complexity": dc})
print("Mux11 XGB:", results["xgboost"]["mux11"])

def _default(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    raise TypeError(o)

json.dump(results, open("complexity_mux11.json", "w"), indent=2, default=_default)
print("Ecrit : complexity_mux11.json")
