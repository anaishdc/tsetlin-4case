"""Baselines classiques (sklearn) sur nos memes fichiers de donnees.

Usage:
    python baseline_classic.py <train.txt> <test.txt> <n_features> [--single-file] [--runs N]
"""
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import BernoulliNB


def load_dataset(path: str, n_features: int):
    data = np.loadtxt(path, dtype=np.float64)
    X = data[:, :n_features]
    y = data[:, n_features].astype(int)
    return X, y


MODELS = {
    "LogReg": lambda seed: LogisticRegression(max_iter=1000, random_state=seed),
    "SVM": lambda seed: SVC(random_state=seed),
    "RandomForest": lambda seed: RandomForestClassifier(n_estimators=200, random_state=seed),
    "MLP": lambda seed: MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=seed),
    "NaiveBayes": lambda seed: BernoulliNB(),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("train")
    parser.add_argument("test")
    parser.add_argument("n_features", type=int)
    parser.add_argument("--single-file", action="store_true")
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed0", type=int, default=20000)
    args = parser.parse_args()

    if args.single_file:
        X_all, y_all = load_dataset(args.train, args.n_features)
    else:
        X_train, y_train = load_dataset(args.train, args.n_features)
        X_test, y_test = load_dataset(args.test, args.n_features)
        print(f"train={len(y_train)} test={len(y_test)} features={args.n_features}")

    results = {name: [] for name in MODELS}

    for run in range(args.runs):
        seed = args.seed0 + run
        if args.single_file:
            rng = np.random.RandomState(seed)
            perm = rng.permutation(len(y_all))
            train_size = int(round(args.train_frac * len(y_all)))
            train_idx, test_idx = perm[:train_size], perm[train_size:]
            X_train, y_train = X_all[train_idx], y_all[train_idx]
            X_test, y_test = X_all[test_idx], y_all[test_idx]

        for name, factory in MODELS.items():
            model = factory(seed)
            model.fit(X_train, y_train)
            acc = 100.0 * model.score(X_test, y_test)
            results[name].append(acc)

    print(f"\n=== Resultats sur {args.runs} runs ===")
    for name, accs in results.items():
        accs = np.array(accs)
        print(f"{name:15s} Mean={accs.mean():.4f} +/- {accs.std():.4f}")


if __name__ == "__main__":
    main()
