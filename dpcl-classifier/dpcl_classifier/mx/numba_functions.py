import numpy as np
from numba import prange
from numba import jit


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
    action_mask_1 = np.zeros((ta_state.shape[0], n, 2), dtype=np.int32)
    action_mask_0 = np.zeros((ta_state.shape[0], n, 2), dtype=np.int32)
    neg_example = 1 - example

    for j in range(ta_state.shape[0]):
        for i in range(n):
            action_result = action(ta_state[j, i, example[i]], states)
            action_mask_1[j, i, 0] = int(action_result == 1)
            action_mask_0[j, i, 0] = int(action_result == 0)

            action_result = action(ta_state[j, i, neg_example[i]], states)
            action_mask_1[j, i, 1] = int(action_result == 1)
            action_mask_0[j, i, 1] = int(action_result == 0)

    return action_mask_1, action_mask_0



@jit(nopython=True)
def update_positive(n, ta_state, example, action_mask_1, action_mask_0, success_mask, success_mask_complement, states):
    neg_example = 1 - example  # Compute the negation of example

    for i in range(n):
        for j in range(ta_state.shape[0]):  # Assuming ta_state has shape (clauses, n, 2)
            ex = example[i]
            neg_ex = neg_example[i]

            # Include with some probability
            if ta_state[j, i, ex] < 2 * states and action_mask_1[j, i, 0]:
                ta_state[j, i, ex] += success_mask[j, 0]
                ta_state[j, i, ex] -= 1 - success_mask[j, 0]

            # Exclude negations
            ta_state[j, i, neg_ex] -= action_mask_1[j, i, 1]

            # Include with some probability
            if ta_state[j, i, ex] > 1 and action_mask_0[j, i, 0]:
                ta_state[j, i, ex] -= success_mask_complement[j, 0]
                ta_state[j, i, ex] += 1 - success_mask_complement[j, 0]

            # Exclude negations
            if ta_state[j, i, neg_ex] > 1 and action_mask_0[j, i, 1]:
                ta_state[j, i, neg_ex] -= action_mask_0[j, i, 1]


@jit(nopython=True)
def update_negative(n, ta_state, example, action_mask_1, action_mask_0, success_mask, success_mask_complement, states):
    neg_example = 1 - example  # Compute the negation of example

    for i in range(n):
        for j in range(ta_state.shape[0]):  # Assuming ta_state has shape (clauses, n, 2)
            ex = example[i]
            neg_ex = neg_example[i]

            # Include with some probability
            if ta_state[j, i, ex] < 2 * states and action_mask_1[j, i, 0]:
                ta_state[j, i, ex] -= 1 - success_mask_complement[j, 0]
                ta_state[j, i, ex] += success_mask_complement[j, 0]

            # Include negations
            if ta_state[j, i, neg_ex] < 2 * states + 1 and action_mask_1[j, i, 1]:
                ta_state[j, i, neg_ex] += success_mask[j, 0]
                ta_state[j, i, neg_ex] -= 1 - success_mask[j, 0]

            # Exclude with some probability
            if ta_state[j, i, ex] > 1 and action_mask_0[j, i, 0]:
                ta_state[j, i, ex] += success_mask_complement[j, 0]
                ta_state[j, i, ex] -= 1 - success_mask_complement[j, 0]

            # Exclude negations
            if ta_state[j, i, neg_ex] > 1 and action_mask_0[j, i, 1]:
                ta_state[j, i, neg_ex] -= success_mask_complement[j, 0]
                ta_state[j, i, neg_ex] += 1 - success_mask_complement[j, 0]
