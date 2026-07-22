"""Copie exacte des fonctions PCL de PCL_BinaryIRIS.py (AAAI-25),
sans l'import keras inutilise qui casse l'import. Aucune ligne de
logique modifiee -- copie literale pour eviter toute erreur de
retranscription.
"""
from numba import jit, prange, njit
import numpy as np

@jit(nopython=True)
def success(prob):
    # return np.random.binomial(1, prob).flatten()
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
            action_result_neg = action(ta_state[j, i, 1- example[i]], states)
            action_mask_1[j, i, example[i]] = int(action_result_ex  == 1)
            action_mask_0[j, i, example[i]] = int(action_result_ex  == 0)

            action_mask_1[j, i, 1- example[i]] = int(action_result_neg == 1)
            action_mask_0[j, i, 1- example[i]] = int(action_result_neg == 0)

    return action_mask_1, action_mask_0


@njit(parallel=True)
def PCL(n, epochs, examples, labels, clauses, states, probabilities):
    ta_state = np.random.choice(np.array([states, states + 1]), size=(clauses, n, 2)).astype(np.int32)
    n_examples = len(examples)
    for _ in range(epochs):
        for e in prange(n_examples):
            example = examples[e]
            # Create masks for positive and negative labels
            positive_labels = (labels[e] == 1)
            negative_labels = (labels[e] == 0)

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
            if ta_state[j, i, ex] < 2 * states and action_mask_1[j, i, ex]:
                ta_state[j, i, ex] += success_mask[j]
                ta_state[j, i, ex] -= (1 - success_mask[j])

            # Exclude negations
            ta_state[j, i, neg_ex] -= action_mask_1[j, i, neg_ex]

            # Include with some probability
            if ta_state[j, i, ex] > 1 and action_mask_0[j, i, ex]:
                ta_state[j, i, ex] -= success_mask_complement[j]
                ta_state[j, i, ex] += (1 - success_mask_complement[j])

            # Exclude negations
            if ta_state[j, i, neg_ex] > 1:
                ta_state[j, i, neg_ex] -= action_mask_0[j, i, neg_ex]


@jit(nopython=True)
def update_negative(n, ta_state, example, action_mask_1, action_mask_0, success_mask, success_mask_complement, states):
    neg_example = 1 - example  # Compute the negation of example

    for i in range(n):
        for j in range(ta_state.shape[0]):  # Assuming ta_state has shape (clauses, n, 2)
            ex = example[i]
            neg_ex = neg_example[i]

            # Include with some probability
            if ta_state[j, i, ex] < 2 * states and action_mask_1[j, i, ex]:
                ta_state[j, i, ex] -= success_mask[j]
                ta_state[j, i, ex] += 1 - success_mask[j]

            # Include negations
            if ta_state[j, i, neg_ex] < 2 * states  and action_mask_1[j, i, neg_ex]:
                ta_state[j, i, neg_ex] += success_mask[j]
                ta_state[j, i, neg_ex] -= 1 - success_mask[j]

            # Exclude with some probability
            if ta_state[j, i, ex] > 1 and action_mask_0[j, i, ex]:
                ta_state[j, i, ex] += success_mask_complement[j]
                ta_state[j, i, ex] -= (1 - success_mask_complement[j])

            # Exclude negations
            if ta_state[j, i, neg_ex] > 1 and action_mask_0[j, i, neg_ex]:
                ta_state[j, i, neg_ex] -= success_mask_complement[j]
                ta_state[j, i, neg_ex] += 1 - success_mask_complement[j]



#@jit(nopython=True)
# def Accuracy(examples, formulas, n, labels):
#     accur = 0
#     examples = np.array(examples)
#     labels = np.array(labels)
#
#     for e in range(len(examples)):
#         allLabels = []
#         # print("------------", e,"---------------")
#         predicted = 0
#         for c in formulas:
#             label = 1
#             for f in range(n):
#                 if c.__contains__((f + 1)) and examples[e][f] == 0:
#                     label = 0
#                     continue
#                 if c.__contains__(-1 * (f + 1)) and examples[e][f] == 1:
#                     label = 0
#                     continue
#             allLabels.append(label)
#         if allLabels.__contains__(1):
#             predicted = 1
#         #if sum(allLabels) > len(formulas) / 2:
#         #    predicted = 1
#         if predicted == labels[e]:
#             accur = accur + 1
#     return accur / len(examples)

#@jit(nopython=True)
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

