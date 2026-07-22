#!/usr/bin/env python3
# -----------------------------------------------------------------
# Compare notre systeme complet (routeur+clauses+AdaBoost, config
# validee) contre XGBoost (arbres+boosting), avec un protocole
# STRICTEMENT SYMETRIQUE au notre :
#   1) UNE recherche d'hyperparametres (une fois, par validation
#      croisee sur le train uniquement -- jamais le test).
#   2) La config trouvee est FIGEE, puis evaluee sur autant de runs
#      que notre propre methode (40 Iris / 20 XOR / 4 Digits), en ne
#      faisant varier que la seed d'entrainement de XGBoost (et le
#      reshuffle train/test pour Iris, comme pour notre systeme).
# Aucune fuite du test set : gs.fit() ne voit jamais Xte/yte.
# -----------------------------------------------------------------
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

PARAM_GRID = {
    "n_estimators": [16, 50, 100, 300],
    "max_depth": [1, 2, 3, 4],
    "learning_rate": [0.1, 0.3],
}

# Notre systeme complet, config validee (K, M), meme nombre de runs
OURS_FINAL = {
    "iris":   (92.3, 4.4, 40),   # K=2, M=16, 40 runs
    "xor":    (100.0, 0.0, 20),  # K=2, M=5,  20 runs
    "digits": (96.6, 0.5, 4),    # K=4, M=16, 4 runs
}

def pick_hyperparams(Xtr, ytr, n_classes):
    """Une seule recherche, par CV sur train uniquement (jamais le test)."""
    objective = "binary:logistic" if n_classes == 2 else "multi:softmax"
    base = XGBClassifier(objective=objective, eval_metric="logloss", n_jobs=4, verbosity=0)
    gs = GridSearchCV(base, PARAM_GRID, cv=3, n_jobs=4, scoring="accuracy")
    gs.fit(Xtr, ytr)
    return gs.best_params_

def run_iris(n_runs=40):
    data = np.loadtxt("../data/BinaryIrisData.txt").astype(int)
    n_features = 16

    # Etape 1 : hyperparametres choisis UNE fois, sur le 1er split (seed 0)
    rng0 = np.random.RandomState(20000)
    d0 = data.copy(); rng0.shuffle(d0)
    split0 = int(round(0.8 * len(d0)))
    params = pick_hyperparams(d0[:split0, :n_features], d0[:split0, n_features], 3)
    print(f"  hyperparametres retenus (1 seule recherche) : {params}")

    # Etape 2 : meme config figee, evaluee sur n_runs splits (comme notre methode)
    accs = []
    for seed in range(n_runs):
        rng = np.random.RandomState(20000 + seed)
        d = data.copy(); rng.shuffle(d)
        split = int(round(0.8 * len(d)))
        Xtr, ytr = d[:split, :n_features], d[:split, n_features]
        Xte, yte = d[split:, :n_features], d[split:, n_features]
        clf = XGBClassifier(objective="multi:softmax", eval_metric="logloss",
                             random_state=seed, **params)
        clf.fit(Xtr, ytr)
        accs.append(100.0 * clf.score(Xte, yte))
    return float(np.mean(accs)), float(np.std(accs)), len(accs)

def run_fixed_split(train_path, test_path, n_features, n_classes, n_runs, objective):
    train = np.loadtxt(train_path).astype(int)
    test  = np.loadtxt(test_path).astype(int)
    Xtr, ytr = train[:, :n_features], train[:, n_features]
    Xte, yte = test[:, :n_features], test[:, n_features]

    params = pick_hyperparams(Xtr, ytr, n_classes)
    print(f"  hyperparametres retenus (1 seule recherche) : {params}")

    accs = []
    for seed in range(n_runs):
        clf = XGBClassifier(objective=objective, eval_metric="logloss",
                             random_state=seed, **params)
        clf.fit(Xtr, ytr)
        accs.append(100.0 * clf.score(Xte, yte))
    return float(np.mean(accs)), float(np.std(accs)), len(accs)

if __name__ == "__main__":
    print("=== XGBoost Iris (40 runs, meme protocole que nous) ===")
    m_iris, s_iris, n_iris = run_iris(40)
    print(f"  -> {m_iris:.1f} +/- {s_iris:.1f} ({n_iris} runs)")

    print("=== XGBoost XOR (20 runs, split fixe, seed d'entrainement variee) ===")
    m_xor, s_xor, n_xor = run_fixed_split(
        "../data/NoisyXORTrainingData.txt", "../data/NoisyXORTestData.txt",
        12, 2, 20, "binary:logistic")
    print(f"  -> {m_xor:.1f} +/- {s_xor:.1f} ({n_xor} runs)")

    print("=== XGBoost Digits (4 runs, split fixe, seed d'entrainement variee) ===")
    m_dig, s_dig, n_dig = run_fixed_split(
        "../data/BinaryDigitsTrainingData192.txt", "../data/BinaryDigitsTestData192.txt",
        192, 10, 4, "multi:softmax")
    print(f"  -> {m_dig:.1f} +/- {s_dig:.1f} ({n_dig} runs)")

    print("\n=== RESUME (meme nombre de runs des deux cotes) ===")
    print(f"{'Dataset':<10} {'Notre systeme':>20} {'XGBoost':>20}")
    for key, (m_o, s_o, n_o), (m_x, s_x, n_x) in [
        ("Iris",   OURS_FINAL["iris"],   (m_iris, s_iris, n_iris)),
        ("XOR",    OURS_FINAL["xor"],    (m_xor, s_xor, n_xor)),
        ("Digits", OURS_FINAL["digits"], (m_dig, s_dig, n_dig)),
    ]:
        print(f"{key:<10} {m_o:>6.1f}% +/- {s_o:<4.1f} ({n_o} runs)   {m_x:>6.1f}% +/- {s_x:<4.1f} ({n_x} runs)")
