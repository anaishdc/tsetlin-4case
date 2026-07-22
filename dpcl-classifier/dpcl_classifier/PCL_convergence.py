# This is a sample Python script.
import argparse
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import itertools
import random
import numpy as np
from loguru import logger
from scipy.stats import bernoulli

import time
from numba import jit

@jit(nopython=True)
def success(prob: float) -> int:
    """Determine success based on a probability value."""
    return 1 if np.random.binomial(1, prob, 1) else 0


@jit(nopython=True)
def action(state: np.ndarray, states: int) -> int:
    """Determine action based on state and states."""
    return 0 if state <= states else 1


@jit(nopython=True)
def PCL(n: int, epochs: int, examples: np.ndarray, labels: np.ndarray,
        clauses: int, states: int, p: float) -> np.ndarray:
    """Perform the Tsetlin Machine Classifier algorithm."""
    ta_state = np.random.choice(np.array([states, states + 1]), size=(clauses, n, 2)).astype(np.int32)

    for i in range(epochs):
        #print('')
        #print("###################################Epoch", i, "###################################")
        #print('')
        for e in range(len(examples)):
            # print("------------------------",e,"------------------")
            # FeedBack on Positives
            if labels[e] == 1:
                for f in range(n):
                    for j in range(clauses):
                        #p = probabilitie[j]
                        # Include with some probability
                        if action(ta_state[j, f, examples[e][f]], states) == 1:
                            if success(p):
                                if ta_state[j, f, examples[e][f]] < 2 * states:
                                    ta_state[j, f, examples[e][f]] = ta_state[j, f, examples[e][f]] + 1
                            else:
                                ta_state[j, f, examples[e][f]] = ta_state[j, f, examples[e][f]] - 1

                        # Exclude Negations
                        if action(ta_state[j, f, int(not examples[e][f])], states) == 1:
                            ta_state[j, f, int(not examples[e][f])] = ta_state[j, f, int(not examples[e][f])] - 1

                        # Include with some probability
                        if action(ta_state[j, f, examples[e][f]], states) == 0:
                            if success(1 - p):
                                if ta_state[j, f, examples[e][f]] > 1:
                                    ta_state[j, f, examples[e][f]] = ta_state[j, f, examples[e][f]] - 1
                            else:
                                ta_state[j, f, examples[e][f]] = ta_state[j, f, examples[e][f]] + 1

                        # Exclude Negations
                        if action(ta_state[j, f, int(not examples[e][f])], states) == 0:
                            if ta_state[j, f, int(not examples[e][f])] > 1:
                                ta_state[j, f, int(not examples[e][f])] = ta_state[j, f, int(not examples[e][f])] - 1

            if labels[e] == 0:
                for f in range(n):
                    for j in range(clauses):
                        #p = probabilitie[j]
                        if action(ta_state[j, f, examples[e][f]], states) == 1:
                            if success(p):
                                ta_state[j, f, examples[e][f]] = ta_state[j, f, examples[e][f]] - 1
                            else:
                                if ta_state[j, f, examples[e][f]] < 2 * states:
                                    ta_state[j, f, examples[e][f]] = ta_state[j, f, examples[e][f]] + 1

                        if action(ta_state[j, f, int(not examples[e][f])], states) == 1:
                            if success(p):
                                if ta_state[j, f, int(not examples[e][f])] < 2 * states + 1:
                                    ta_state[j, f, int(not examples[e][f])] = ta_state[j, f, int(not examples[e][f])] + 1
                            else:
                                ta_state[j, f, int(not examples[e][f])] = ta_state[j, f, int(not examples[e][f])] - 1

                        if action(ta_state[j, f, examples[e][f]], states) == 0:
                            if success(1 - p):
                                ta_state[j, f, examples[e][f]] = ta_state[j, f, examples[e][f]] + 1

                            else:
                                if ta_state[j, f, examples[e][f]] > 1:
                                    ta_state[j, f, examples[e][f]] = ta_state[j, f, examples[e][f]] - 1

                        if action(ta_state[j, f, int(not examples[e][f])], states) == 0:
                            if success(1 - p):
                                if ta_state[j, f, int(not examples[e][f])] > 1:
                                    ta_state[j, f, int(not examples[e][f])] = ta_state[j, f, int(not examples[e][f])] - 1
                            else:
                                ta_state[j, f, int(not examples[e][f])] = ta_state[j, f, int(not examples[e][f])] + 1

    return ta_state


def main(n: int, states: int, epochs: int, clauses: int, runs: int, p: float) -> None:
    start_all = time.time()
    #examples = list(map(list, itertools.product([0, 1], repeat=n)))
    suc, fails = 0, []

    for run_number in range(runs):
        start = time.process_time()
        examples = list(map(list, itertools.product([0, 1], repeat=n)))
        # logger.info("------------- Run {} -----------------------------", run_number)

        target = np.random.choice(3, n, replace=True)
        flag = True
        for i in target:
            if (i != 2):
                flag = False
                break
        if flag:
            target[0] = 1
        labels = [1] * len(examples)

        for X in examples:
            if any((x == 1 and t == 0) or (x == 0 and t == 1) for x, t in zip(X, target)):
                labels[examples.index(X)] = 0

        s = ""
        t = []
        for i in range(len(target)):
            if target[i] == 1:
                s += f"X_{i+1} and "
                t.append(i+1)
            if target[i] == 0:
                s += f"not X_{i + 1} and "
                t.append(-i - 1)

        # logger.info("Target: {}", s[:-5])  # removed the last " and "
        # logger.info("Target List: {}", t)

        result = PCL(n, epochs, np.array(examples), np.array(labels), clauses, states, p)

        formulas = []
        for i in range(clauses):
            c = [j + 1 if result[i, j, 1] > states else -j - 1 for j in range(n) if result[i, j, 0] > states or result[i, j, 1] > states]
            formulas.append(c)

        # logger.info("Learned: {}", formulas)

        if t in formulas:
            suc += 1
        else:
            fails.append(t)
        #
        # logger.info("Contains Target: {}", t in formulas)
        # logger.info("Time for this run: {:.2f} seconds", time.process_time() - start)

    logger.info("Successes: {}/{}", suc, runs)
    logger.info("Fails: {}", fails)
    logger.info("Total Time: {:.2f} seconds", time.time() - start_all)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Execute Tsetlin Machine Classifier Algorithm.')
    parser.add_argument('--n', type=int, default=2, help='Number of elements.')
    parser.add_argument('--states', type=int, default=10000, help='Number of states.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs.')
    parser.add_argument('--clauses', type=int, default=1, help='Number of clauses.')
    parser.add_argument('--runs', type=int, default=100, help='Number of runs.')
    parser.add_argument('--p', type=float, default=0.75, help='Probability value.')

    args = parser.parse_args()

    logger.info(f"Executing with arguments: {args}")
    for i in range(1, 13):
        logger.info("------------- N = {} ---------------------", i)

        main(i, args.states, 1000, args.clauses, args.runs, args.p)
