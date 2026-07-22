#!/usr/bin/env python3
# -----------------------------------------------------------------
# Compare notre systeme (routeur+clauses+AdaBoost) et notre baseline
# (routeur seul+AdaBoost) contre un arbre de decision isole (CART,
# sklearn), en fonction de K (nombre de feuilles), sur Iris et XOR
# (Digits ajoute des que son balayage C++ est termine).
#
# But : isoler ce que les clauses apportent par rapport a un arbre
# seul (pas boost), a nombre de feuilles egal.
# -----------------------------------------------------------------
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# ---- Resultats deja mesures (notre C++, systeme complet + baseline) ----
# (K, mean_systeme, std_systeme, mean_baseline, std_baseline)
OURS = {
    "iris": [
        (2, 93.1, 4.5, 52.7, 23.9),
        (4, 94.0, 3.5, 92.0, 4.7),
        (8, 95.1, 4.4, 95.3, 3.8),
    ],
    "xor": [
        (2, 100.0, 0.0, 50.1, 0.0),
        (4, 85.4, 7.2, 44.8, 12.0),
        (8, 69.1, 8.2, 46.8, 20.5),
    ],
}

def cart_iris(K_values, n_seeds=15):
    data = np.loadtxt("../data/BinaryIrisData.txt").astype(int)
    n_features = 16
    out = {}
    for K in K_values:
        accs = []
        for seed in range(n_seeds):
            rng = np.random.RandomState(20000 + seed)
            d = data.copy(); rng.shuffle(d)
            split = int(round(0.8 * len(d)))
            Xtr, ytr = d[:split, :n_features], d[:split, n_features]
            Xte, yte = d[split:, :n_features], d[split:, n_features]
            clf = DecisionTreeClassifier(max_leaf_nodes=max(2, K), random_state=seed)
            clf.fit(Xtr, ytr)
            accs.append(100.0 * clf.score(Xte, yte))
        out[K] = (float(np.mean(accs)), float(np.std(accs)))
    return out

def cart_xor(K_values):
    train = np.loadtxt("../data/NoisyXORTrainingData.txt").astype(int)
    test  = np.loadtxt("../data/NoisyXORTestData.txt").astype(int)
    n_features = 12
    Xtr, ytr = train[:, :n_features], train[:, n_features]
    Xte, yte = test[:, :n_features], test[:, n_features]
    out = {}
    for K in K_values:
        clf = DecisionTreeClassifier(max_leaf_nodes=max(2, K), random_state=0)
        clf.fit(Xtr, ytr)
        out[K] = (100.0 * clf.score(Xte, yte), 0.0)
    return out

def cart_digits(K_values):
    Xtr_all = np.loadtxt("../data/BinaryDigitsTrainingData192.txt").astype(int)
    Xte_all = np.loadtxt("../data/BinaryDigitsTestData192.txt").astype(int)
    n_features = 192
    Xtr, ytr = Xtr_all[:, :n_features], Xtr_all[:, n_features]
    Xte, yte = Xte_all[:, :n_features], Xte_all[:, n_features]
    out = {}
    for K in K_values:
        clf = DecisionTreeClassifier(max_leaf_nodes=max(2, K), random_state=0)
        clf.fit(Xtr, ytr)
        out[K] = (100.0 * clf.score(Xte, yte), 0.0)
    return out

if __name__ == "__main__":
    import json
    results = {}
    print("=== CART Iris ===")
    r = cart_iris([2, 4, 8])
    for K, (m, s) in r.items(): print(f"K={K}: {m:.1f} +/- {s:.1f}")
    results["iris"] = r

    print("=== CART XOR ===")
    r = cart_xor([2, 4, 8])
    for K, (m, s) in r.items(): print(f"K={K}: {m:.1f}")
    results["xor"] = r

    print("=== CART Digits ===")
    r = cart_digits([2, 4, 8, 16, 32])
    for K, (m, s) in r.items(): print(f"K={K}: {m:.1f}")
    results["digits"] = r

    with open("cart_results.json", "w") as f:
        json.dump({d: {str(k): v for k, v in r.items()} for d, r in results.items()}, f, indent=2)
    print("\nEcrit dans cart_results.json")
