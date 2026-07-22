from keras.datasets import mnist
from numba import jit, prange, njit
import time
import numpy as np
from loguru import logger


@jit(nopython=True)
def success(prob):
    # return np.random.binomial(1, prob).flatten()
    N = prob.shape[0]
    accepted = np.zeros((N, 1), dtype=np.int32)
    for i in prange(N):
        accepted[i] = 1 if np.random.random() < prob[i] else 0
    return accepted


@jit(nopython=True)
def action(state, states):
    return np.int32(state > states)


@jit(nopython=True)
def create_action_masks(ta_state, example, states, n):
    action_mask_1 = np.zeros((ta_state.shape[0], n), dtype=np.int32)
    action_mask_0 = np.zeros((ta_state.shape[0], n), dtype=np.int32)

    for j in range(ta_state.shape[0]):
        for i in range(n):
            action_result = action(ta_state[j, i, example[i]], states)
            action_mask_1[j, i] = int(action_result == 1)
            action_mask_0[j, i] = int(action_result == 0)

    return action_mask_1, action_mask_0


@njit(parallel=True)
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
            # action_mask_1 = action(ta_state[:, np.arange(n), example], states) == 1
            # action_mask_0 = action(ta_state[:, np.arange(n), example], states) == 0
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


@jit(nopython=True)
def update_positive(n, ta_state, example, action_mask_1, action_mask_0, success_mask, success_mask_complement, states):
    neg_example = 1 - example  # Compute the negation of example

    for i in range(n):
        for j in range(ta_state.shape[0]):  # Assuming ta_state has shape (clauses, n, 2)
            ex = example[i]
            neg_ex = neg_example[i]

            # Include with some probability
            if ta_state[j, i, ex] < 2 * states and action_mask_1[j, i]:
                ta_state[j, i, ex] += success_mask[j, 0]
                ta_state[j, i, ex] -= 1 - success_mask[j, 0]

            # Exclude negations
            ta_state[j, i, neg_ex] -= action_mask_1[j, i]

            # Include with some probability
            if ta_state[j, i, ex] > 1 and action_mask_0[j, i]:
                ta_state[j, i, ex] -= success_mask_complement[j, 0]
                ta_state[j, i, ex] += 1 - success_mask_complement[j, 0]

            # Exclude negations
            if ta_state[j, i, neg_ex] > 1:
                ta_state[j, i, neg_ex] -= action_mask_0[j, i]


@jit(nopython=True)
def update_negative(n, ta_state, example, action_mask_1, action_mask_0, success_mask, success_mask_complement, states):
    neg_example = 1 - example  # Compute the negation of example

    for i in range(n):
        for j in range(ta_state.shape[0]):  # Assuming ta_state has shape (clauses, n, 2)
            ex = example[i]
            neg_ex = neg_example[i]

            # Include with some probability
            if ta_state[j, i, ex] < 2 * states and action_mask_1[j, i]:
                ta_state[j, i, ex] -= success_mask[j, 0]
                ta_state[j, i, ex] += 1 - success_mask[j, 0]

            # Include negations
            if ta_state[j, i, neg_ex] < 2 * states + 1 and action_mask_1[j, i]:
                ta_state[j, i, neg_ex] += success_mask[j, 0]
                ta_state[j, i, neg_ex] -= 1 - success_mask[j, 0]

            # Exclude with some probability
            if ta_state[j, i, ex] > 1 and action_mask_0[j, i]:
                ta_state[j, i, ex] += success_mask_complement[j, 0]
                ta_state[j, i, ex] -= 1 - success_mask_complement[j, 0]

            # Exclude negations
            if ta_state[j, i, neg_ex] > 1:
                ta_state[j, i, neg_ex] -= success_mask_complement[j, 0]
                ta_state[j, i, neg_ex] += 1 - success_mask_complement[j, 0]


def Accuracy(examples, formulas, n, labels):
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
            neg_labels =  [True for i in range(len(examples))]
            neg_labels = np.array(neg_labels)

        all_labels[:, c_idx] = pos_labels & neg_labels

    predicted = (all_labels.sum(axis=1) > 0).astype(int)
    accuracy = (predicted == labels).mean()

    return accuracy


def evaluate_model(result, states, clauses, n, X_test_filtered, Y_test_filtered):

    # Extract formulas from the result
    #formulas = [[j + 1 if result[i, j, 1] > states else -j - 1 if result[i, j, 0] > states for j in range(n) if
                 #result[i, j, 1] > states or result[i, j, 0] > states] for i in range(clauses)]

    formulas = []
    for i in range(clauses):
        c = []
        for j in range(n):
            if (result[i, j, 1] > states):
                c.append(j + 1)
            if (result[i, j, 0] > states):
                c.append(-j - 1)
        formulas.append(c)

    accuracy = Accuracy(X_test_filtered, formulas, n, Y_test_filtered)

    return accuracy

def run_experiment(states=10000, epochs=1, clauses=150, runs=100, pl=0.6, pu=0.8):
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    X_train = np.where(X_train.reshape((X_train.shape[0], 28 * 28)) > 75, 1, 0)
    X_test = np.where(X_test.reshape((X_test.shape[0], 28 * 28)) > 75, 1, 0)

    digit1 = 0
    digit2 = 1

    mask_train = np.isin(Y_train, [digit1, digit2])
    X_train_filtered = X_train[mask_train]
    Y_train_filtered = Y_train[mask_train]
    Y_train_filtered[Y_train_filtered == digit1] = 0
    Y_train_filtered[Y_train_filtered == digit2] = 1

    mask_test = np.isin(Y_test, [digit1, digit2])
    X_test_filtered = X_test[mask_test]
    Y_test_filtered = Y_test[mask_test]
    Y_test_filtered[Y_test_filtered == digit1] = 0
    Y_test_filtered[Y_test_filtered == digit2] = 1

    Accuracies = []

    for i in range(runs):
        logger.info("-------- Run {} --------", i)
        probs = np.random.uniform(pl, pu, clauses)

        start = time.time()
        n = X_train_filtered.shape[1]

        result = TMC(n, epochs, X_train_filtered, Y_train_filtered, clauses, states, probs)
        accuracy = evaluate_model(result, states, clauses, n, X_test_filtered, Y_test_filtered)
        Accuracies.append(accuracy)

        logger.info("Time = {}", time.time() - start)
        logger.info("Accuracy: {}", accuracy)


    logger.info("Accuracies: {}", Accuracies)
    logger.info("MIN: {}", min(Accuracies))
    logger.info("MAX: {}", max(Accuracies))
    logger.info("AVG: {}", np.mean(Accuracies))
    logger.info("STD: {}", np.std(Accuracies))



def main():

    states = 10000
    epochs = 10
    clauses = 1
    runs = 1
    pl = 0.5
    pu = 0.55

    run_experiment(states, epochs, clauses, runs, pl, pu)



if __name__ == '__main__':
    main()
