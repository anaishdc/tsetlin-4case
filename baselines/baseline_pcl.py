"""Baseline officiel : Probabilistic Concept Learning (cair/dpcl-classifier, papier AAAI-25).

Les fonctions numba ci-dessous (action/success/create_action_masks/PCL/
update_positive/update_negative) sont copiees telles quelles depuis
Main_code_AAAI25/PCL_BinaryIRIS.py, le script de demo officiel du papier
AAAI-25 (pas depuis dpcl_classifier/classifier_numba_mc.py : ce dernier
contient un bug confirme -- positive_labels = labels[e] == target, avec
target := labels[e], est donc toujours vrai, si bien que la branche
"negative" du one-vs-rest n'est jamais executee -- et son evaluate_model
compare des pixels binaires a l'indice de classe c via
examples[:, pos_features] == c, ce qui n'a de sens que pour c in {0,1}).

PCL_BinaryIRIS.py lui-meme ne traite que la classification binaire
(Y_train[Y_train != 1] = 0, une classe contre le reste). On reconstruit
donc le multiclasse nous-memes, en one-vs-rest, au-dessus de ce
classifieur binaire officiel et correct.

Usage:
    python baseline_pcl.py <train.txt> <test.txt> <n_features> [options]
"""
import argparse
import time
import numpy as np
from numba import jit, prange, njit


# ---------------------------------------------------------------------------
# Copie verbatim de Main_code_AAAI25/PCL_BinaryIRIS.py (sans le
# "from keras.datasets import mnist" du fichier d'origine, non utilise ici,
# et sans la partie matplotlib/plotting).
# ---------------------------------------------------------------------------

@jit(nopython=True)
def success(prob):
    N = prob.shape[0]
    accepted = np.zeros(N, dtype=np.int32)
    for i in prange(N):
        accepted[i] = 1 if np.random.random() < prob[i] else 0
    return accepted


@jit(nopython=True)
def action(state, states):
    return np.int32(state > states)


@jit(nopython=True)
def create_action_masks(ta_state, example, states, n):
    action_mask_1 = np.zeros((ta_state.shape[0], n, 2), dtype=np.int32)
    action_mask_0 = np.zeros((ta_state.shape[0], n, 2), dtype=np.int32)

    for j in range(ta_state.shape[0]):
        for i in range(n):
            action_result_ex = action(ta_state[j, i, example[i]], states)
            action_result_neg = action(ta_state[j, i, 1 - example[i]], states)
            action_mask_1[j, i, example[i]] = int(action_result_ex == 1)
            action_mask_0[j, i, example[i]] = int(action_result_ex == 0)
            action_mask_1[j, i, 1 - example[i]] = int(action_result_neg == 1)
            action_mask_0[j, i, 1 - example[i]] = int(action_result_neg == 0)

    return action_mask_1, action_mask_0


@jit(nopython=True)
def update_positive(n, ta_state, example, action_mask_1, action_mask_0, success_mask, success_mask_complement, states):
    neg_example = 1 - example
    for i in range(n):
        for j in range(ta_state.shape[0]):
            ex = example[i]
            neg_ex = neg_example[i]
            if ta_state[j, i, ex] < 2 * states and action_mask_1[j, i, ex]:
                ta_state[j, i, ex] += success_mask[j]
                ta_state[j, i, ex] -= (1 - success_mask[j])
            ta_state[j, i, neg_ex] -= action_mask_1[j, i, neg_ex]
            if ta_state[j, i, ex] > 1 and action_mask_0[j, i, ex]:
                ta_state[j, i, ex] -= success_mask_complement[j]
                ta_state[j, i, ex] += (1 - success_mask_complement[j])
            if ta_state[j, i, neg_ex] > 1:
                ta_state[j, i, neg_ex] -= action_mask_0[j, i, neg_ex]


@jit(nopython=True)
def update_negative(n, ta_state, example, action_mask_1, action_mask_0, success_mask, success_mask_complement, states):
    neg_example = 1 - example
    for i in range(n):
        for j in range(ta_state.shape[0]):
            ex = example[i]
            neg_ex = neg_example[i]
            if ta_state[j, i, ex] < 2 * states and action_mask_1[j, i, ex]:
                ta_state[j, i, ex] -= success_mask[j]
                ta_state[j, i, ex] += 1 - success_mask[j]
            if ta_state[j, i, neg_ex] < 2 * states and action_mask_1[j, i, neg_ex]:
                ta_state[j, i, neg_ex] += success_mask[j]
                ta_state[j, i, neg_ex] -= 1 - success_mask[j]
            if ta_state[j, i, ex] > 1 and action_mask_0[j, i, ex]:
                ta_state[j, i, ex] += success_mask_complement[j]
                ta_state[j, i, ex] -= (1 - success_mask_complement[j])
            if ta_state[j, i, neg_ex] > 1 and action_mask_0[j, i, neg_ex]:
                ta_state[j, i, neg_ex] -= success_mask_complement[j]
                ta_state[j, i, neg_ex] += 1 - success_mask_complement[j]


@njit(parallel=True)
def PCL(n, epochs, examples, labels, clauses, states, probabilities):
    ta_state = np.random.choice(np.array([states, states + 1]), size=(clauses, n, 2)).astype(np.int32)
    n_examples = len(examples)
    for _ in range(epochs):
        for e in prange(n_examples):
            example = examples[e]
            positive_labels = (labels[e] == 1)
            negative_labels = (labels[e] == 0)

            action_mask_1, action_mask_0 = create_action_masks(ta_state, example, states, n)
            p = probabilities[:, np.newaxis]
            success_mask = success(p)
            success_mask_complement = success(1 - p)

            if positive_labels:
                update_positive(n, ta_state, example, action_mask_1, action_mask_0, success_mask,
                                 success_mask_complement, states)
            elif negative_labels:
                update_negative(n, ta_state, example, action_mask_1, action_mask_0, success_mask,
                                 success_mask_complement, states)

    return ta_state


def extract_formulas(ta_state, states, clauses, n):
    formulas = []
    for i in range(clauses):
        c = []
        for j in range(n):
            if ta_state[i, j, 1] > states:
                c.append(j + 1)
            if ta_state[i, j, 0] > states:
                c.append(-j - 1)
        formulas.append(c)
    return formulas


def clause_scores(examples, formulas):
    """Nombre de clauses (conjonctions PCL) qui matchent chaque exemple (OR de conjonctions)."""
    examples = np.asarray(examples)
    all_labels = np.zeros((len(examples), len(formulas)), dtype=bool)
    for c_idx, c in enumerate(formulas):
        pos_features = np.array([f for f in c if f > 0], dtype=np.int64) - 1
        neg_features = np.array([-f for f in c if f < 0], dtype=np.int64) - 1

        pos_labels = (examples[:, pos_features] == 1).all(axis=1) if pos_features.size else np.ones(len(examples), dtype=bool)
        neg_labels = (examples[:, neg_features] == 0).all(axis=1) if neg_features.size else np.ones(len(examples), dtype=bool)
        all_labels[:, c_idx] = pos_labels & neg_labels
    return all_labels.sum(axis=1)


# ---------------------------------------------------------------------------
# Wrapper one-vs-rest (multiclasse), construit au-dessus du classifieur binaire.
# ---------------------------------------------------------------------------

def load_dataset(path: str, n_features: int):
    data = np.loadtxt(path, dtype=np.uint32)
    if data.ndim != 2 or data.shape[1] != n_features + 1:
        raise ValueError(f"{path}: attendu {n_features + 1} colonnes, recu {data.shape}")
    X = np.ascontiguousarray(data[:, :n_features], dtype=np.int32)
    y = np.ascontiguousarray(data[:, n_features], dtype=np.int32)
    return X, y


def train_and_eval_one_vs_rest(X_train, y_train, X_test, y_test, n_classes, clauses, states, epochs, pl, pu):
    n = X_train.shape[1]
    scores_per_class = np.zeros((len(y_test), n_classes))

    for c in range(n_classes):
        y_bin = (y_train == c).astype(np.int32)
        probs = np.random.uniform(pl, pu, clauses)
        ta_state = PCL(n, epochs, X_train, y_bin, clauses, states, probs)
        formulas = extract_formulas(ta_state, states, clauses, n)
        scores_per_class[:, c] = clause_scores(X_test, formulas)

    predicted = scores_per_class.argmax(axis=1)
    return 100.0 * np.mean(predicted == y_test)


def train_and_eval_binary(X_train, y_train, X_test, y_test, clauses, states, epochs, pl, pu):
    n = X_train.shape[1]
    probs = np.random.uniform(pl, pu, clauses)
    ta_state = PCL(n, epochs, X_train, y_train.astype(np.int32), clauses, states, probs)
    formulas = extract_formulas(ta_state, states, clauses, n)
    scores = clause_scores(X_test, formulas)
    predicted = (scores > 0).astype(int)
    return 100.0 * np.mean(predicted == y_test)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("train")
    parser.add_argument("test")
    parser.add_argument("n_features", type=int)
    parser.add_argument("--n-classes", type=int, default=2, help="2 = binaire direct (XOR) ; >2 = one-vs-rest")
    parser.add_argument("--clauses", type=int, default=16)
    parser.add_argument("--states", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--pl", type=float, default=0.6)
    parser.add_argument("--pu", type=float, default=0.8)
    parser.add_argument("--seed0", type=int, default=20000)
    parser.add_argument("--single-file", action="store_true")
    parser.add_argument("--train-frac", type=float, default=0.8)
    args = parser.parse_args()

    if args.single_file:
        X_all, y_all = load_dataset(args.train, args.n_features)
    else:
        X_train, y_train = load_dataset(args.train, args.n_features)
        X_test, y_test = load_dataset(args.test, args.n_features)
        print(f"train={len(y_train)} test={len(y_test)} features={args.n_features}")

    print(f"n_classes={args.n_classes} clauses={args.clauses} states={args.states} "
          f"epochs={args.epochs} pl={args.pl} pu={args.pu} runs={args.runs}")

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

        start = time.perf_counter()
        if args.n_classes <= 2:
            acc = train_and_eval_binary(X_train, y_train, X_test, y_test,
                                         args.clauses, args.states, args.epochs, args.pl, args.pu)
        else:
            acc = train_and_eval_one_vs_rest(X_train, y_train, X_test, y_test, args.n_classes,
                                              args.clauses, args.states, args.epochs, args.pl, args.pu)
        elapsed = time.perf_counter() - start

        accuracies.append(acc)
        times.append(elapsed)
        print(f"  run {run} : acc={acc:.4f}%  time={elapsed:.2f}s")

    accuracies = np.asarray(accuracies)
    times = np.asarray(times)
    print(f"PCL : Mean={accuracies.mean():.4f} +/- {accuracies.std():.4f}  time_mean={times.mean():.2f}s")


if __name__ == "__main__":
    main()
