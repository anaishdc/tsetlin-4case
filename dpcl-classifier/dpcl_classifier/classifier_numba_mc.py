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


@jit(nopython=True)
def pcl_update_positive(n, ta_state, example, action_mask_1, action_mask_0, success_mask, success_mask_complement,
                        states):
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
def pcl_update_negative(n, ta_state, example, action_mask_1, action_mask_0, success_mask, success_mask_complement,
                        states):
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


@njit(parallel=True)
def pcl_fit(
        ta_state,
        n_classes,
        n_state_bits,
        n_literals,
        probabilities,
        epochs,
        examples,
        labels
):
    n_examples = len(examples)

    for _ in range(epochs):
        for e in prange(n_examples):
            example = examples[e]
            # Create masks for positive and negative labels
            target = labels[e]
            not_target = target
            while not_target == target:
                not_target = np.random.randint(0, n_classes)

            # Create masks for actions
            # Compute probabilities and success
            action_mask_1, action_mask_0 = create_action_masks(ta_state[target], example, n_state_bits, n_literals)
            success_mask = success(probabilities[target])
            success_mask_complement = success(1 - probabilities[target])
            
            positive_labels = labels[e] == target
            negative_labels = labels[e] != target
        

            # Update ta_state for positive labels
            if positive_labels:
                pcl_update_positive(
                    n_literals,
                    ta_state[target],
                    example,
                    action_mask_1,
                    action_mask_0,
                    success_mask,
                    success_mask_complement,
                    n_state_bits
                )


            elif negative_labels:
                pcl_update_negative(
                    n_literals,
                    ta_state[not_target],
                    example,
                    action_mask_1,
                    action_mask_0,
                    success_mask,
                    success_mask_complement,
                    n_state_bits
                )

    return ta_state


class PCL:

    def __init__(
            self,
            pl,
            pu,
            n_classes,
            n_clauses,
            n_state_bits,
            n_literals
    ):
        self.pl = pl
        self.pu = pu
        self.n_clauses = n_clauses
        self.n_state_bits = n_state_bits
        self.n_literals = n_literals
        self.n_classes = n_classes

        self.probabilities = np.random.uniform(self.pl, self.pu, (n_classes, n_clauses, 1))
        self.ta_state = np.random.choice(
            np.array([self.n_state_bits, self.n_state_bits + 1]), size=(n_classes, n_clauses, self.n_literals, 2)
        ).astype(np.int32)

    def fit(self, epochs, examples, labels):
        self.ta_state = pcl_fit(
            self.ta_state,
            self.n_classes,
            self.n_state_bits,
            self.n_literals,
            self.probabilities,
            epochs,
            examples,
            labels
        )

    def evaluate_model(self, X_test_filtered, Y_test_filtered):
        examples = X_test_filtered
        labels = Y_test_filtered

        positive_mask = self.ta_state[:, :, :, 0] > self.n_state_bits
        negative_mask = self.ta_state[:, :, :, 1] > self.n_state_bits

        num_classes = self.n_classes
        num_clauses = self.n_clauses
        num_examples = X_test_filtered.shape[0]

        # Initialize arrays to hold the pos_labels and neg_labels
        labels_all = np.zeros((num_examples, num_classes, num_clauses))

        for c in range(num_classes):
            for cl in range(num_clauses):
                # Extract the positive and negative masks for the current class and clause
                positive_mask_current = positive_mask[c, cl, :]
                negative_mask_current = negative_mask[c, cl, :]

                # Find the feature indices where the masks are true
                pos_features = np.where(positive_mask_current)[0]
                neg_features = np.where(negative_mask_current)[0]

                # Calculate the pos_labels and neg_labels for the current class and clause
                if pos_features.size != 0:
                    pos_labels = (examples[:, pos_features] == c).all(axis=1)
                else:
                    pos_labels = [True for i in range(len(examples))]
                    pos_labels = np.array(pos_labels)

                if neg_features.size != 0:
                    neg_labels = (examples[:, neg_features] != c).all(axis=1)
                else:
                    neg_labels = [True for i in range(len(examples))]
                    neg_labels = np.array(neg_labels)

                # Store the pos_labels and neg_labels in the arrays
                labels_all[:, c, cl] = pos_labels & neg_labels

        # Now pos_labels_all and neg_labels_all contain the pos_labels and neg_labels for all classes and all clauses
        class_sums = labels_all.sum(axis=2)
        predicted = class_sums.argmax(axis=1)
        print(list(predicted))
        accuracy = (predicted == labels).mean()

        return accuracy


def run_experiment(states=10000, epochs=1, clauses=150, runs=100, pl=0.6, pu=0.8):
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    X_train = np.where(X_train.reshape((X_train.shape[0], 28 * 28)) > 75, 1, 0)
    X_test = np.where(X_test.reshape((X_test.shape[0], 28 * 28)) > 75, 1, 0)

    mask_train = np.isin(Y_train, [0, 1])
    X_train_filtered = X_train[mask_train]
    Y_train_filtered = Y_train[mask_train]

    mask_test = np.isin(Y_test, [0, 1])
    X_test_filtered = X_test[mask_test]
    Y_test_filtered = Y_test[mask_test]

    Accuracies = []

    pcl = PCL(
        pl=pl,
        pu=pu,
        n_classes=2,  # TODO
        n_clauses=clauses,
        n_state_bits=states,
        n_literals=X_train_filtered.shape[1]
    )

    import pickle
    for i in range(runs):
        logger.info("-------- Run {} --------", i)

        start = time.time()

        pcl.fit(epochs, X_train_filtered, Y_train_filtered)
        accuracy = pcl.evaluate_model(X_test_filtered, Y_test_filtered)

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
    epochs = 1
    clauses = 50
    runs = 100
    pl = 0.25
    pu = 0.35

    run_experiment(states, epochs, clauses, runs, pl, pu)


if __name__ == '__main__':
    main()
