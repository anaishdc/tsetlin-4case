import argparse
import itertools
import random
import time
from typing import List, Tuple

import numpy as np
from loguru import logger
from numba import jit


@jit(nopython=True)
def check_success(prob: np.ndarray) -> np.ndarray:
    """Check the success of an event based on a given probability.

    Args:
        prob (float): The probability of success.

    Returns:
        int: 1 if the event is successful, 0 otherwise.
    """
    return np.random.binomial(1, prob, 1)


@jit(nopython=True)
def determine_action(state: np.ndarray, states: int) -> int:
    """Determine the action based on the state.

    Args:
        state (int): The current state.
        states (int): The number of states.

    Returns:
        int: 0 if state is less than or equal to states, else 1.
    """
    return 0 if state <= states else 1


@jit(nopython=True)
def perform_tmc(n: int, epochs: int, examples: np.ndarray, labels: np.ndarray, clauses: int,
                states: int, probabilities: np.ndarray) -> np.ndarray:
    """Perform TMC and return the state of Tsetlin Automata.

    Args:
        n (int): Number of features.
        epochs (int): Number of training epochs.
        examples (np.ndarray): Array of examples.
        labels (np.ndarray): Array of labels.
        clauses (int): Number of clauses.
        states (int): Number of states.
        probabilities (np.ndarray): Array of probabilities for each clause.

    Returns:
        np.ndarray: The state of the Tsetlin Automata.
    """
    ta_state = np.random.choice(np.array([states, states + 1]), size=(clauses, n, 2)).astype(np.int32)

    for i in range(epochs):
        # print('')
        # print("###################################Epoch", i, "###################################")
        # print('')
        for e in range(len(examples)):
            # print("------------------------",e,"------------------")
            # FeedBack on Positives
            if labels[e] == 1:
                for f in range(n):
                    for j in range(clauses):
                        p = probabilities[j]
                        # Include with some probability
                        if determine_action(ta_state[j, f, examples[e][f]], states) == 1:
                            if check_success(p):
                                if ta_state[j, f, examples[e][f]] < 2 * states:
                                    ta_state[j, f, examples[e][f]] = ta_state[j, f, examples[e][f]] + 1
                            else:
                                ta_state[j, f, examples[e][f]] = ta_state[j, f, examples[e][f]] - 1

                        # Exclude Negations
                        if determine_action(ta_state[j, f, int(not examples[e][f])], states) == 1:
                            ta_state[j, f, int(not examples[e][f])] = ta_state[j, f, int(not examples[e][f])] - 1

                        # Include with some probability
                        if determine_action(ta_state[j, f, examples[e][f]], states) == 0:
                            if check_success(1 - p):
                                if ta_state[j, f, examples[e][f]] > 1:
                                    ta_state[j, f, examples[e][f]] = ta_state[j, f, examples[e][f]] - 1
                            else:
                                ta_state[j, f, examples[e][f]] = ta_state[j, f, examples[e][f]] + 1

                        # Exclude Negations
                        if determine_action(ta_state[j, f, int(not examples[e][f])], states) == 0:
                            if ta_state[j, f, int(not examples[e][f])] > 1:
                                ta_state[j, f, int(not examples[e][f])] = ta_state[j, f, int(not examples[e][f])] - 1

            if labels[e] == 0:
                for f in range(n):
                    for j in range(clauses):
                        p = probabilities[j]
                        if determine_action(ta_state[j, f, examples[e][f]], states) == 1:
                            if check_success(p):
                                ta_state[j, f, examples[e][f]] = ta_state[j, f, examples[e][f]] - 1
                            else:
                                if ta_state[j, f, examples[e][f]] < 2 * states:
                                    ta_state[j, f, examples[e][f]] = ta_state[j, f, examples[e][f]] + 1

                        if determine_action(ta_state[j, f, int(not examples[e][f])], states) == 1:
                            if check_success(p):
                                if ta_state[j, f, int(not examples[e][f])] < 2 * states + 1:
                                    ta_state[j, f, int(not examples[e][f])] = ta_state[
                                                                                  j, f, int(not examples[e][f])] + 1
                            else:
                                ta_state[j, f, int(not examples[e][f])] = ta_state[j, f, int(not examples[e][f])] - 1

                        if determine_action(ta_state[j, f, examples[e][f]], states) == 0:
                            if check_success(1 - p):
                                ta_state[j, f, examples[e][f]] = ta_state[j, f, examples[e][f]] + 1

                            else:
                                if ta_state[j, f, examples[e][f]] > 1:
                                    ta_state[j, f, examples[e][f]] = ta_state[j, f, examples[e][f]] - 1

                        if determine_action(ta_state[j, f, int(not examples[e][f])], states) == 0:
                            if check_success(1 - p):
                                if ta_state[j, f, int(not examples[e][f])] > 1:
                                    ta_state[j, f, int(not examples[e][f])] = ta_state[
                                                                                  j, f, int(not examples[e][f])] - 1
                            else:
                                ta_state[j, f, int(not examples[e][f])] = ta_state[j, f, int(not examples[e][f])] + 1

    return ta_state


# @jit(nopython=True)
# Function to calculate accuracy
def calculate_accuracy(examples: List[List[int]], formulas: List[List[int]], n: int, labels: List[int]) -> float:
    """Calculate the accuracy of the learned formulas.

    Args:
        examples (List[List[int]]): List of examples.
        formulas (List[List[int]]): List of learned formulas.
        n (int): Number of features.
        labels (List[int]): List of correct labels.

    Returns:
        float: The accuracy of the learned formulas.
    """
    accur = 0

    for e in range(len(examples)):
        allLabels = []
        # print("------------", e,"---------------")
        predicted = 0
        for c in formulas:
            label = 1
            for f in range(n):
                if c.__contains__((f + 1)) and examples[e][f] == 0:
                    label = 0
                    continue
                if c.__contains__(-1 * (f + 1)) and examples[e][f] == 1:
                    label = 0
                    continue
            allLabels.append(label)
        if allLabels.__contains__(1):
            predicted = 1
        # if sum(allLabels) > len(formulas) / 2:
        #    predicted = 1
        if predicted == labels[e]:
            accur = accur + 1
    return accur / len(examples)


def main(args):
    start_all = time.process_time()
    num_features = args.num_features
    examples = list(map(list, itertools.product([0, 1], repeat=num_features)))
    target = args.target
    labels = [1 if all(x == target_i for x, target_i in zip(example, target)) else 0 for example in examples]

    accuracies = []
    for i in range(args.runs):
        logger.info(f"----- Run {i} --------")
        X_training, y_training = np.array(examples), np.array(labels)
        X_test, y_test = X_training, y_training
        logger.info(f"Labels Counts: {np.unique(y_test, return_counts=True)}")

        probabilities = np.array([random.uniform(args.lower_prob, args.upper_prob) for _ in range(args.clauses)])
        logger.info(f"Probabilities: {probabilities}")

        start = time.process_time()
        result = perform_tmc(num_features, args.epochs, X_training, y_training, args.clauses, args.states, probabilities)

        formulas = [[(j + 1) if result[i, j, 1] > args.states else -(j + 1) for j in range(num_features)] for i in
                    range(args.clauses)]

        logger.info(f"Learned Formulas: {formulas}")
        logger.info(f"Time Taken: {time.process_time() - start} seconds")

        accuracy = calculate_accuracy(X_test, formulas, num_features, y_test)
        accuracies.append(accuracy)
        logger.info(f"Accuracy: {accuracy}")

    logger.info(f"Accuracies: {accuracies}")
    logger.info(f"MIN Accuracy: {min(accuracies)}")
    logger.info(f"MAX Accuracy: {max(accuracies)}")
    logger.info(f"AVG Accuracy: {sum(accuracies) / len(accuracies)}")
    logger.info(f"STD Accuracy: {np.std(accuracies)}")
    logger.info(f"Total Time Taken: {time.process_time() - start_all} seconds")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform TMC.')
    parser.add_argument('--num_features', type=int, default=10, help='Number of features')
    parser.add_argument('--target', type=int, nargs='+', default=[0, 1, 2, 0, 0, 1, 2, 2, 0, 1], help='Target list')
    parser.add_argument('--states', type=int, default=10000, help='Number of states')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--clauses', type=int, default=1, help='Number of clauses')
    parser.add_argument('--runs', type=int, default=1, help='Number of runs')
    parser.add_argument('--lower_prob', type=float, default=0.6, help='Lower bound for probability')
    parser.add_argument('--upper_prob', type=float, default=0.8, help='Upper bound for probability')

    args = parser.parse_args()
    main(args)
