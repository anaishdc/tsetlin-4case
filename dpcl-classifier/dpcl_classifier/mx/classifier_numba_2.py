from keras.datasets import mnist
from numba import jit, prange, njit
import time
import numpy as np
from loguru import logger
from tqdm import tqdm

from dpcl_classifier.mx.numba_functions import create_action_masks, success, update_positive, update_negative


@njit(parallel=True)
def tmc_trainer(target, n, examples, labels, ta_state, states, probabilities):
    # ta_state = np.random.choice(np.array([states, states + 1]), size=(clauses, n, 2)).astype(np.int32)
    n_examples = len(examples)


    for e in prange(n_examples):
        example = examples[e]
        # Create masks for positive and negative labels
        positive_labels = labels[e] == target
        negative_labels = labels[e] != target

        # Create masks for actions
        action_mask_1, action_mask_0 = create_action_masks(ta_state, example, states, n)

        # Compute probabilities and success
        p = probabilities[:, np.newaxis]
        success_mask = success(p)
        success_mask_complement = success(1 - p)

        # Update ta_state for positive labels
        if positive_labels:
            update_positive(n, ta_state, example, action_mask_1, action_mask_0, success_mask,
                            success_mask_complement, states)
        # Update ta_state for negative labels
        elif negative_labels:
            update_negative(n, ta_state, example, action_mask_1, action_mask_0, success_mask,
                            success_mask_complement, states)

    return ta_state


class TMCTarget:

    def __init__(self, target, n_states, n_clauses, pl, pu):
        self.target = target
        self.n_clauses = n_clauses
        self.n_states = n_states
        self.pl = pl
        self.pu = pu
        self.ta_state = np.random.choice(np.array([n_states, n_states + 1]), size=(n_clauses, n_states, 2)).astype(
            np.int32)
        self.probabilities = np.random.uniform(pl, pu, n_clauses)


class TMC:

    def __init__(self, X, y, n_states, n_clauses, epochs, pl, pu):
        self.X = X
        self.y = y
        self.n_states = n_states
        self.n_clauses = n_clauses
        self.epochs = epochs
        self.pl = pl
        self.pu = pu
        self.classes = np.unique(y)
        self.n_samples = len(X)
        self.n_features = X.shape[1]
        self.classes_data = {k: TMCTarget(k, n_states, n_clauses, pl, pu) for k in self.classes}

    def fit(self, X_test, y_test):

        for epoch in tqdm(range(self.epochs)):
            class_sums = np.zeros((max(self.classes) + 1, self.n_samples))

            for target, tm in self.classes_data.items():
                tm.ta_state = tmc_trainer(
                    target,
                    self.n_features,
                    self.X,
                    self.y,
                    tm.ta_state,
                    tm.n_states,
                    tm.probabilities
                )

                class_sum = evaluate_model(
                    target,
                    tm.ta_state,
                    tm.n_states,
                    tm.n_clauses,
                    self.n_features,
                    X_test,
                    y_test,
                )
                class_sums[target] = class_sum

            predicted = np.argmax(class_sums, axis=0)
            accuracy = (predicted == y_test).mean()
            logger.info("Epoch={} - Accuracy: {}", epoch, accuracy)

        print(":D")


def compute_class_sum(target, examples, formulas, n, labels):
    examples = np.array(examples)
    labels = np.array(labels)
    all_labels = np.zeros((len(examples), len(formulas)))

    for c_idx, c in enumerate(formulas):
        pos_features = np.array([f for f in c if f > 0]) - 1
        neg_features = np.array([-f for f in c if f < 0]) - 1

        if pos_features.size != 0:
            pos_labels = (examples[:, pos_features] == 1).all(axis=1)
        else:
            pos_labels = [True for i in range(len(examples))]
            pos_labels = np.array(pos_labels)

        if neg_features.size != 0:
            neg_labels = (examples[:, neg_features] == 0).all(axis=1)
        else:
            neg_labels = [True for i in range(len(examples))]
            neg_labels = np.array(neg_labels)

        all_labels[:, c_idx] = pos_labels & neg_labels

    class_sum = all_labels.sum(axis=1)
    # predicted = (class_sum > 0).astype(int)

    # Get indexes where predicted is 1
    # predicted_idx = np.where(predicted == 1)[0]

    # Get indexes where predicted is 0
    # predicted_idx_complement = np.where(predicted == 0)[0]

    # target_predict = labels[predicted_idx] == target
    # not_target_predict = labels[predicted_idx_complement] != target

    # Calculate accuracy
    # accuracy = (target_predict.sum() + not_target_predict.sum()) / len(examples)

    # Set predicted to "target" if predicted is 1, otherwise set it to the other class
    # predicted = np.where(predicted == 1, target, -1)

    # accuracy = (predicted == labels).mean()

    return class_sum


def evaluate_model(target, result, states, clauses, n, X_test_filtered, Y_test_filtered):
    # Extract formulas from the result
    # formulas = [[j + 1 if result[i, j, 1] > states else -j - 1 if result[i, j, 0] > states for j in range(n) if
    # result[i, j, 1] > states or result[i, j, 0] > states] for i in range(clauses)]

    formulas = []
    for i in range(clauses):
        c = []
        for j in range(n):
            if result[i, j, 1] > states:
                c.append(j + 1)
            if result[i, j, 0] > states:
                c.append(-j - 1)
        formulas.append(c)

    class_sum = compute_class_sum(target, X_test_filtered, formulas, n, Y_test_filtered)

    return class_sum


def run_experiment(states=10000, epochs=1, clauses=150, runs=100, pl=0.6, pu=0.8):


    # Load data
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    classes = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    # Binarize data
    X_train = np.where(X_train.reshape((X_train.shape[0], 28 * 28)) > 75, 1, 0)
    X_test = np.where(X_test.reshape((X_test.shape[0], 28 * 28)) > 75, 1, 0)

    N = 100  # Number of samples per class

    X_train_filtered = []
    Y_train_filtered = []

    X_test_filtered = []
    Y_test_filtered = []

    for c in classes:
        # Find indices for each class in the training set
        indices_train = np.where(Y_train == c)[0]
        chosen_indices_train = np.random.choice(indices_train, N, replace=False)
        X_train_filtered.append(X_train[chosen_indices_train])
        Y_train_filtered.append(Y_train[chosen_indices_train])

        # Find indices for each class in the test set
        indices_test = np.where(Y_test == c)[0]
        chosen_indices_test = np.random.choice(indices_test, N, replace=False)
        X_test_filtered.append(X_test[chosen_indices_test])
        Y_test_filtered.append(Y_test[chosen_indices_test])

    # Convert lists to numpy arrays
    X_train_filtered = np.concatenate(X_train_filtered, axis=0)
    Y_train_filtered = np.concatenate(Y_train_filtered, axis=0)
    X_test_filtered = np.concatenate(X_test_filtered, axis=0)
    Y_test_filtered = np.concatenate(Y_test_filtered, axis=0)



    x = TMC(
        X=X_train_filtered,
        y=Y_train_filtered,
        n_states=states,
        n_clauses=clauses,
        epochs=epochs,
        pl=pl,
        pu=pu
    )

    x.fit(
        X_test=X_test_filtered,
        y_test=Y_test_filtered
    )


def main():
    states = 10000
    epochs = 100
    clauses = 500
    runs = 1
    pl = 0.5
    pu = 0.55

    run_experiment(states, epochs, clauses, runs, pl, pu)


if __name__ == '__main__':
    main()
