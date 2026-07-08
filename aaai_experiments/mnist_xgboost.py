#!/usr/bin/env python3
# XGBoost sur MNIST (784 bits binarises, 60k train / 10k test).
# Grille reduite (cv=2) vu la taille du dataset, pour rester dans un temps raisonnable.
import time
import numpy as np
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

PARAM_GRID = {
    "n_estimators": [100, 300],
    "max_depth": [4, 6],
    "learning_rate": [0.1, 0.3],
}

print("chargement...", flush=True)
train = np.loadtxt("../data/MNISTTrainingOfficial60k.txt").astype(int)
test  = np.loadtxt("../data/MNISTTestOfficial10k.txt").astype(int)
n = 784
Xtr, ytr = train[:, :n], train[:, n]
Xte, yte = test[:, :n], test[:, n]
print("shapes", Xtr.shape, Xte.shape, flush=True)

t0 = time.time()
base = XGBClassifier(objective="multi:softmax", num_class=10, eval_metric="mlogloss", n_jobs=4, verbosity=1)
gs = GridSearchCV(base, PARAM_GRID, cv=2, n_jobs=1, scoring="accuracy", verbose=2)
gs.fit(Xtr, ytr)
t1 = time.time()
acc = 100.0 * gs.best_estimator_.score(Xte, yte)
print(f"MNIST XGBoost best: {acc:.1f}%  params={gs.best_params_}  temps_recherche={t1-t0:.1f}s", flush=True)

import json
json.dump({"mnist": acc, "params": gs.best_params_}, open("mnist_xgboost.json", "w"))
