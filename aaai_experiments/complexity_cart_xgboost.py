#!/usr/bin/env python3
# -----------------------------------------------------------------
# Complexite (nb de conditions / decision complexity par prediction)
# pour CART (plusieurs niveaux d'elagage) et XGBoost (plusieurs
# n_estimators), sur Iris/XOR/Digits -- pour tracer accuracy vs
# log(complexite) contre notre systeme.
# -----------------------------------------------------------------
import json
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

def cart_complexity(clf, X):
    n_internal = clf.tree_.node_count - clf.get_n_leaves()
    paths = clf.decision_path(X)  # sparse, nb de noeuds visites (incl. feuille) par exemple
    avg_depth = paths.sum(axis=1).mean() - 1  # -1 pour exclure la feuille elle-meme
    return n_internal, float(avg_depth)

def xgb_complexity(clf, n_estimators, max_depth):
    df = clf.get_booster().trees_to_dataframe()
    n_internal = int((df["Feature"] != "Leaf").sum())
    # decision complexity : n_estimators * profondeur moyenne reelle des arbres
    depths = []
    for tid in df["Tree"].unique():
        sub = df[df["Tree"] == tid]
        # profondeur = nb de '-' dans l'ID (format "0-1-0" style xgboost) -- approx via longueur du Node id split
        depths.append(sub["Feature"].ne("Leaf").sum())  # approx : nb de splits internes de cet arbre
    avg_depth_per_tree = np.mean(depths) if depths else max_depth
    return n_internal, n_estimators * avg_depth_per_tree

# ---- Iris (reshuffle 80/20 comme notre systeme, seed=0 pour un point representatif) ----
def load_iris_split(seed=0):
    data = np.loadtxt("../data/BinaryIrisData.txt").astype(int)
    rng = np.random.RandomState(20000 + seed)
    d = data.copy(); rng.shuffle(d)
    split = int(round(0.8 * len(d)))
    return d[:split, :16], d[:split, 16], d[split:, :16], d[split:, 16]

def load_fixed(train_path, test_path, n):
    tr = np.loadtxt(train_path).astype(int)
    te = np.loadtxt(test_path).astype(int)
    return tr[:, :n], tr[:, n], te[:, :n], te[:, n]

results = {"cart": {}, "xgboost": {}}

# --- Iris ---
Xtr, ytr, Xte, yte = load_iris_split(0)
results["cart"]["iris"] = []
for K in [2, 4, 8, 16, 32]:
    clf = DecisionTreeClassifier(max_leaf_nodes=K, random_state=0).fit(Xtr, ytr)
    n_int, depth = cart_complexity(clf, Xte)
    acc = 100.0 * clf.score(Xte, yte)
    results["cart"]["iris"].append({"K": K, "acc": acc, "total_complexity": n_int, "decision_complexity": depth})
print("Iris CART:", results["cart"]["iris"])

results["xgboost"]["iris"] = []
for ne in [1, 4, 16, 50, 100, 300]:
    clf = XGBClassifier(objective="multi:softmax", n_estimators=ne, max_depth=2, learning_rate=0.1,
                         eval_metric="mlogloss", n_jobs=4, verbosity=0).fit(Xtr, ytr)
    n_int, dc = xgb_complexity(clf, ne, 2)
    acc = 100.0 * clf.score(Xte, yte)
    results["xgboost"]["iris"].append({"n_estimators": ne, "acc": acc, "total_complexity": n_int, "decision_complexity": dc})
print("Iris XGB:", results["xgboost"]["iris"])

# --- XOR ---
Xtr, ytr, Xte, yte = load_fixed("../data/NoisyXORTrainingData.txt", "../data/NoisyXORTestData.txt", 12)
results["cart"]["xor"] = []
for K in [2, 4, 8, 16, 32]:
    clf = DecisionTreeClassifier(max_leaf_nodes=K, random_state=0).fit(Xtr, ytr)
    n_int, depth = cart_complexity(clf, Xte)
    acc = 100.0 * clf.score(Xte, yte)
    results["cart"]["xor"].append({"K": K, "acc": acc, "total_complexity": n_int, "decision_complexity": depth})
print("XOR CART:", results["cart"]["xor"])

results["xgboost"]["xor"] = []
for ne in [1, 4, 16, 50, 100, 300]:
    clf = XGBClassifier(objective="binary:logistic", n_estimators=ne, max_depth=2, learning_rate=0.1,
                         eval_metric="logloss", n_jobs=4, verbosity=0).fit(Xtr, ytr)
    n_int, dc = xgb_complexity(clf, ne, 2)
    acc = 100.0 * clf.score(Xte, yte)
    results["xgboost"]["xor"].append({"n_estimators": ne, "acc": acc, "total_complexity": n_int, "decision_complexity": dc})
print("XOR XGB:", results["xgboost"]["xor"])

# --- Digits ---
Xtr, ytr, Xte, yte = load_fixed("../data/BinaryDigitsTrainingData192.txt", "../data/BinaryDigitsTestData192.txt", 192)
results["cart"]["digits"] = []
for K in [2, 4, 8, 16, 32, 64]:
    clf = DecisionTreeClassifier(max_leaf_nodes=K, random_state=0).fit(Xtr, ytr)
    n_int, depth = cart_complexity(clf, Xte)
    acc = 100.0 * clf.score(Xte, yte)
    results["cart"]["digits"].append({"K": K, "acc": acc, "total_complexity": n_int, "decision_complexity": depth})
print("Digits CART:", results["cart"]["digits"])

results["xgboost"]["digits"] = []
for ne in [1, 4, 16, 50, 100, 300]:
    clf = XGBClassifier(objective="multi:softmax", n_estimators=ne, max_depth=3, learning_rate=0.3,
                         eval_metric="mlogloss", n_jobs=4, verbosity=0).fit(Xtr, ytr)
    n_int, dc = xgb_complexity(clf, ne, 3)
    acc = 100.0 * clf.score(Xte, yte)
    results["xgboost"]["digits"].append({"n_estimators": ne, "acc": acc, "total_complexity": n_int, "decision_complexity": dc})
print("Digits XGB:", results["xgboost"]["digits"])

def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    raise TypeError(o)

json.dump(results, open("complexity_results.json", "w"), indent=2, default=_default)
print("Ecrit : complexity_results.json")
