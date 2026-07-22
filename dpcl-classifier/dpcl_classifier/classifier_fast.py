import random
import numpy as np
from tqdm import tqdm
from keras.datasets import mnist
import time
from numba import jit, config, njit, prange
import time
import numpy as np
from loguru import logger



def succcess(prob):
    return np.random.binomial(1, prob)


@jit(nopython=True)
def action(state, states):
    return (state > states).astype(np.int32)



def TMC(n, epochs, examples, labels, clauses, states, probabilities):
    ta_state = np.random.choice(np.array([states, states + 1]), size=(clauses, n, 2)).astype(np.int32)
    n_examples = len(examples)

    for _ in range(epochs):
        for e in prange(n_examples):
            example = examples[e]
            # Create masks for positive and negative labels
            positive_labels = labels[e] == 1
            negative_labels = labels[e] == 0

            # Create masks for actions
            action_mask_1 = action(ta_state[:, np.arange(n), example], states) == 1
            action_mask_0 = action(ta_state[:, np.arange(n), example], states) == 0

            # Compute probabilities and success
            p = probabilities[:, np.newaxis]
            success_mask = succcess(p)
            success_mask_complement = succcess(1 - p)

            # Update ta_state for positive labels
            if positive_labels:
                update_positive(n, ta_state, example, action_mask_1, action_mask_0, success_mask,
                                success_mask_complement, states)
            # Update ta_state for negative labels
            elif negative_labels:
                update_negative(n, ta_state, example, action_mask_1, action_mask_0, success_mask,
                                success_mask_complement, states)

    return ta_state



def update_positive(n, ta_state, example, action_mask_1, action_mask_0, success_mask, success_mask_complement, states):
    neg_example = 1 - example  # Compute the negation of example

    # Include with some probability
    include_condition = (ta_state[:, np.arange(n), example] < 2 * states) & action_mask_1
    ta_state[:, np.arange(n), example] += include_condition & success_mask
    ta_state[:, np.arange(n), example] -= include_condition & ~success_mask

    # Exclude negations
    ta_state[:, np.arange(n), neg_example] -= action_mask_1

    # Include with some probability
    include_condition = (ta_state[:, np.arange(n), example] > 1) & action_mask_0
    ta_state[:, np.arange(n), example] -= include_condition & success_mask_complement
    ta_state[:, np.arange(n), example] += include_condition & ~success_mask_complement

    # Exclude negations
    exclude_condition = ta_state[:, np.arange(n), neg_example] > 1
    ta_state[:, np.arange(n), neg_example] -= exclude_condition & action_mask_0



def update_negative(n, ta_state, example, action_mask_1, action_mask_0, success_mask, success_mask_complement, states):
    neg_example = 1 - example  # Compute the negation of example

    # Include with some probability
    include_condition = (ta_state[:, np.arange(n), example] < 2 * states) & action_mask_1
    ta_state[:, np.arange(n), example] -= include_condition & success_mask
    ta_state[:, np.arange(n), example] += include_condition & ~success_mask

    # Include negations
    include_negation_condition = (ta_state[:, np.arange(n), neg_example] < 2 * states + 1) & action_mask_1
    ta_state[:, np.arange(n), neg_example] += include_negation_condition & success_mask
    ta_state[:, np.arange(n), neg_example] -= include_negation_condition & ~success_mask

    # Exclude with some probability
    exclude_condition = (ta_state[:, np.arange(n), example] > 1) & action_mask_0
    ta_state[:, np.arange(n), example] += exclude_condition & success_mask_complement
    ta_state[:, np.arange(n), example] -= exclude_condition & ~success_mask_complement

    # Exclude negations
    exclude_negation_condition = (ta_state[:, np.arange(n), neg_example] > 1) & action_mask_0
    ta_state[:, np.arange(n), neg_example] -= exclude_negation_condition & success_mask_complement
    ta_state[:, np.arange(n), neg_example] += exclude_negation_condition & ~success_mask_complement


def Accuracy(examples, formulas, n, labels):
    examples = np.array(examples)
    labels = np.array(labels)
    all_labels = np.zeros((len(examples), len(formulas)))

    for c_idx, c in enumerate(formulas):
        pos_features = np.array([f for f in c if f > 0]) - 1
        neg_features = np.array([-f for f in c if f < 0]) - 1

        pos_labels = (examples[:, pos_features] == 1).all(axis=1)
        neg_labels = (examples[:, neg_features] == 0).all(axis=1)

        all_labels[:, c_idx] = pos_labels & neg_labels

    predicted = (all_labels.sum(axis=1) > 0).astype(int)
    accuracy = (predicted == labels).mean()

    return accuracy


def evaluate_model(result, states, clauses, n, X_test_filtered, Y_test_filtered):
    start_time = time.process_time()

    # Extract formulas from the result
    formulas = [[j + 1 if result[i, j, 1] > states else -j - 1 for j in range(n) if
                 result[i, j, 1] > states or result[i, j, 0] > states] for i in range(clauses)]

    logger.info("Time = {}", time.process_time() - start_time)
    accuracy = Accuracy(X_test_filtered, formulas, n, Y_test_filtered)
    logger.info("Accuracy: {}", accuracy)

    return accuracy


def main():
    startAll = time.process_time()
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    X_train = np.where(X_train.reshape((X_train.shape[0], 28 * 28)) > 75, 1, 0)
    X_test = np.where(X_test.reshape((X_test.shape[0], 28 * 28)) > 75, 1, 0)

    mask_train = np.isin(Y_train, [0, 1])
    X_train_filtered = X_train[mask_train]
    Y_train_filtered = Y_train[mask_train]

    mask_test = np.isin(Y_test, [0, 1])
    X_test_filtered = X_test[mask_test]
    Y_test_filtered = Y_test[mask_test]

    states = 10000
    epochs = 1
    clauses = 500
    runs = 100
    pl = 0.6
    pu = 0.8
    Accuracies = []

    for i in range(runs):
        logger.info("-------- Run {} --------", i)
        epochs = i + 1
        probs = np.random.uniform(pl, pu, clauses)

        start = time.process_time()
        n = X_train_filtered.shape[1]

        result = TMC(n, epochs, X_train_filtered, Y_train_filtered, clauses, states, probs)

        formulas = [[j + 1 if result[i, j, 1] > states else -j - 1 for j in range(n) if
                     result[i, j, 1] > states or result[i, j, 0] > states] for i in range(clauses)]

        logger.info("Time = {}", time.process_time() - start)
        accuracy = Accuracy(X_test_filtered, formulas, n, Y_test_filtered)
        Accuracies.append(accuracy)
        logger.info("Accuracy: {}", accuracy)

    logger.info("Accuracies: {}", Accuracies)
    logger.info("MIN: {}", min(Accuracies))
    logger.info("MAX: {}", max(Accuracies))
    logger.info("AVG: {}", np.mean(Accuracies))
    logger.info("STD: {}", np.std(Accuracies))


if __name__ == '__main__':
    main()
