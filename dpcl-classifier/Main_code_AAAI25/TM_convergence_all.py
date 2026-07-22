from tmu.models.classification.vanilla_classifier import TMClassifier
import numpy as np
from time import time
import itertools

number_of_features = 6
noise = 0.0

epochs = 100

s = 1.1
clauses = 1
T = 3

#examples = 2**number_of_features*epochs

for run_n in range(4, 13):
    number_of_features = run_n
    print("------------- N=", run_n)

    errors = 0

    for e in range(100):
        examples = list(map(list, itertools.product([0, 1], repeat=number_of_features)))

        target = np.random.choice(3, number_of_features, replace=True)
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
        examples = np.array(examples).astype(np.uint32)
        labels = np.array(labels).astype(np.uint32)

        tm = TMClassifier(clauses*2, T, s, type_i_ii_ratio=1.0, platform='CPU', boost_true_positive_feedback=0)

        for epoch in range(epochs):
           tm.fit(examples, labels)

        accuracy = 100*(tm.predict(examples) == labels).mean()

        #print("Accuracy:", 100*(tm.predict(examples) == labels).mean())

        #np.set_printoptions(threshold=np.inf, linewidth=200, precision=2, suppress=True)

        precision = tm.clause_precision(1, 0, examples, labels)
        recall = tm.clause_recall(1, 0, examples, labels)

        if precision != 1.0 or recall != 1.0:
            errors += 1
            #print("ERROR")

        # for j in range(clauses):
        #     print("Clause #%d W:%d P:%.2f R:%.2f " % (j, tm.get_weight(1, 0, j), precision[j], recall[j]))
        #     l = []
        #     for k in range(number_of_features*2):
        #         if tm.get_ta_action(j, k, the_class = 1, polarity = 0):
        #             if k < number_of_features:
        #                 print("\tINCLUDE: x%d (TA State %d)" % (k, tm.get_ta_state(j, k, the_class = 1, polarity = 0)))
        #             else:
        #                 print("\tINCLUDE ¬x%d (TA State %d)" % (k-number_of_features, tm.get_ta_state(j, k, the_class = 1, polarity = 0)))
        #         else:
        #             if k < number_of_features:
        #                 print("\tEXCLUDE: x%d (TA State %d)" % (k, tm.get_ta_state(j, k, the_class = 1, polarity = 0)))
        #             else:
        #                 print("\tEXCLUDE ¬x%d (TA State %d)" % (k-number_of_features, tm.get_ta_state(j, k, the_class = 1, polarity = 0)))
    print("Number of successes: ", 100 - errors)
