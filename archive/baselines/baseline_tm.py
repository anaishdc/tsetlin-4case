"""Baseline officiel : Tsetlin Machine (cair/pyTsetlinMachine), MultiClassTsetlinMachine.

Usage:
    python baseline_tm.py <train.txt> <test.txt> <n_features> [options]

Format des fichiers : une ligne par exemple, n_features colonnes binaires
puis le label en derniere colonne (identique au format utilise par
modele_xor.cpp / modele_multiclasse.cpp).
"""
import argparse
import time
import numpy as np

from pyTsetlinMachine.tm import MultiClassTsetlinMachine


def load_dataset(path: str, n_features: int):
    data = np.loadtxt(path, dtype=np.uint32)
    if data.ndim != 2 or data.shape[1] != n_features + 1:
        raise ValueError(f"{path}: attendu {n_features + 1} colonnes, recu {data.shape}")
    X = np.ascontiguousarray(data[:, :n_features], dtype=np.uint32)
    y = np.ascontiguousarray(data[:, n_features], dtype=np.uint32)
    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("train")
    parser.add_argument("test")
    parser.add_argument("n_features", type=int)
    parser.add_argument("--clauses", type=int, default=1000)
    parser.add_argument("--T", type=int, default=50)
    parser.add_argument("--s", type=float, default=10.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--boost-true-positive-feedback", type=int, default=0)
    parser.add_argument("--state-bits", type=int, default=8, help="2^state_bits = nombre d'etats par automate")
    parser.add_argument("--seed0", type=int, default=20000)
    parser.add_argument(
        "--single-file",
        action="store_true",
        help="train == test : reshuffle et re-splitte 80/20 a chaque run "
             "(reproduit exactement le protocole Iris de modele_multiclasse.cpp)",
    )
    parser.add_argument("--train-frac", type=float, default=0.8)
    args = parser.parse_args()

    if args.single_file:
        X_all, y_all = load_dataset(args.train, args.n_features)
    else:
        X_train, y_train = load_dataset(args.train, args.n_features)
        X_test, y_test = load_dataset(args.test, args.n_features)
        print(f"train={len(y_train)} test={len(y_test)} features={args.n_features}")

    print(f"clauses={args.clauses} T={args.T} s={args.s} epochs={args.epochs} runs={args.runs}")

    accuracies = []
    times = []

    for run in range(args.runs):
        np.random.seed(args.seed0 + run)

        if args.single_file:
            perm = np.random.RandomState(args.seed0 + run).permutation(len(y_all))
            train_size = int(round(args.train_frac * len(y_all)))
            train_idx, test_idx = perm[:train_size], perm[train_size:]
            X_train, y_train = X_all[train_idx], y_all[train_idx]
            X_test, y_test = X_all[test_idx], y_all[test_idx]

        model = MultiClassTsetlinMachine(
            args.clauses,
            args.T,
            args.s,
            boost_true_positive_feedback=args.boost_true_positive_feedback,
            number_of_state_bits=args.state_bits,
        )

        start = time.perf_counter()
        model.fit(X_train, y_train, epochs=args.epochs)
        elapsed = time.perf_counter() - start

        pred = model.predict(X_test)
        accuracy = 100.0 * np.mean(pred == y_test)

        accuracies.append(accuracy)
        times.append(elapsed)
        print(f"  run {run} : acc={accuracy:.4f}%  time={elapsed:.2f}s")

    accuracies = np.asarray(accuracies)
    times = np.asarray(times)
    print(f"TM : Mean={accuracies.mean():.4f} +/- {accuracies.std():.4f}  time_mean={times.mean():.2f}s")


if __name__ == "__main__":
    main()
