import random
import numpy as np
from keras.datasets import mnist
import time
from numba import jit


'''
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = np.where(X_train.reshape((X_train.shape[0], 28 * 28)) > 75, 1, 0)
X_test = np.where(X_test.reshape((X_test.shape[0], 28 * 28)) > 75, 1, 0)

# Convert the data to pandas DataFrames for easier sampling
df_X = pd.DataFrame(X_train)
df_Y = pd.DataFrame(Y_train, columns=['label'])

# Combine the X and Y DataFrames
df = pd.concat([df_X, df_Y], axis=1)

# Initialize an empty DataFrame for the final balanced dataset
balanced_data = pd.DataFrame()

# Loop through each class
for i in range(10):
    # Get a subset of the data for the current class
    class_subset = df[df['label'] == i]

    # Sample 1000 instances from the current class
    class_sample = class_subset.sample(n=1000, random_state=42)

    # Append the sample to the final dataset
    balanced_data = pd.concat([balanced_data, class_sample], axis=0)

# Shuffle the final dataset
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Split the balanced dataset back into data and target
X_train_1k = balanced_data.drop('label', axis=1).values
Y_train_1k = balanced_data['label'].values

Y_train_1k[Y_train_1k != 1] = 0
Y_test[Y_test != 1] = 0

print(X_train_1k.shape)
print(Y_train_1k.shape)
'''


@jit(nopython=True)
def success(prob):
    if np.random.binomial(1, prob, 1):
        # if (bernoulli.rvs(prob) == 1):
        return 1
    else:
        return 0


@jit(nopython=True)
def action(state, states):
    if state <= states:
        return 0
    else:
        return 1




@jit(nopython=True)
def TMC(n, epochs, examples, labels, clauses, states, probs):
    ta_state = np.random.choice(np.array([states, states + 1]), size=(clauses, n, 2)).astype(np.int32)
    n_examples = len(examples)

    for _ in range(epochs):
        # print('')
        # print("###################################Epoch", i, "###################################")
        # print('')
        for e in range(n_examples):
            # print("------------------------",e,"------------------")
            # FeedBack on Positives
            for f in range(n):
                for j in range(clauses):

                    if labels[e] == 1:
                        p = probs[j]
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

                    elif labels[e] == 0:

                        p = probs[j]
                        if action(ta_state[j, f, examples[e][f]], states) == 1:
                            if success(p):
                                ta_state[j, f, examples[e][f]] = ta_state[j, f, examples[e][f]] - 1
                            else:
                                if ta_state[j, f, examples[e][f]] < 2 * states:
                                    ta_state[j, f, examples[e][f]] = ta_state[j, f, examples[e][f]] + 1

                        if action(ta_state[j, f, int(not examples[e][f])], states) == 1:
                            if success(p):
                                if ta_state[j, f, int(not examples[e][f])] < 2 * states + 1:
                                    ta_state[j, f, int(not examples[e][f])] = ta_state[
                                                                                  j, f, int(not examples[e][f])] + 1
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
                                    ta_state[j, f, int(not examples[e][f])] = ta_state[
                                                                                  j, f, int(not examples[e][f])] - 1
                            else:
                                ta_state[j, f, int(not examples[e][f])] = ta_state[j, f, int(not examples[e][f])] + 1

    return ta_state


def Accuracy(examples, formulas, n, labels):
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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Use a breakpoint in the code line below to debug your script.
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
    #epochs = 1
    clauses = 10
    runs = 100
    # pl < pu
    pl = 0.6
    pu = 0.8
    Accuracies = []
    for i in range(runs):
        epochs = i + 1
        print("-----Run", i, "--------")

        probs = [random.uniform(pl, pu) for i in range(clauses)]
        probs = np.array(probs)
        # probs = [0.9]
        # print(probs)

        start = time.process_time()
        n = X_train_filtered.shape[1]

        result = TMC(n, epochs, X_train_filtered, Y_train_filtered, clauses, states, probs)
        # print(result)

        formulas = []
        for i in range(clauses):
            c = []
            for j in range(n):
                if result[i, j, 1] > states:
                    c.append(j + 1)
                if result[i, j, 0] > states:
                    c.append(-j - 1)
            formulas.append(c)
        # print("Learned: ", formulas)

        print("Time = ", time.process_time() - start)
        a = Accuracy(X_test_filtered, formulas, n, Y_test_filtered)
        Accuracies.append(a)
        print("Accuracy: ", a)
    print(Accuracies)
    print("MIN: ", min(Accuracies))
    print("MAX: ", max(Accuracies))
    print("AVG: ", sum(Accuracies) / len(Accuracies))
    print("STD: ", np.std(Accuracies))
