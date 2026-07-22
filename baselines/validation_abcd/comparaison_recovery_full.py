"""Protocole "a" de la Table 7 de PCL (AAAI-25) : toute la table de verite a
chaque epoch, 100 epochs. Reutilise leur code officiel tel quel pour PCL-a
(dpcl-classifier/Main_code_AAAI25/PCL_convergence_all.py) et pour TM-a (meme
fichier TM_convergence_all.py, bibliotheque tmu, s=1.1, T=3, 1 clause) ;
ajoute seulement notre regle avec exactement le meme protocole. Complement
direct de comparaison_recovery_balanced.py (protocole "b").

Usage: python3 comparaison_recovery_full.py
"""
import itertools
import sys
import time

import numpy as np
from numba import njit

sys.path.insert(0, "../dpcl-classifier/Main_code_AAAI25")
from PCL_convergence_all import PCL as PCL_run  # noqa: E402

from tmu.models.classification.vanilla_classifier import TMClassifier  # noqa: E402


@njit(cache=True)
def notre_regle(n, epochs, examples, labels, states, a, b, c, d):
    ta_state = np.random.choice(np.array([states, states + 1]), size=(n, 2)).astype(np.int64)
    for _ in range(epochs):
        for e in range(len(examples)):
            y = labels[e]
            for f in range(n):
                xf = examples[e, f]
                v = ta_state[f, 1]
                vbar = ta_state[f, 0]
                if xf == 0 and y == 0:
                    if np.random.random() < a and v < 2 * states:
                        v += 1
                    if np.random.random() < c and vbar > 1:
                        vbar -= 1
                elif xf == 0 and y == 1:
                    if np.random.random() < b and v > 1:
                        v -= 1
                    if np.random.random() < d and vbar < 2 * states:
                        vbar += 1
                elif xf == 1 and y == 0:
                    if np.random.random() < c and v > 1:
                        v -= 1
                    if np.random.random() < a and vbar < 2 * states:
                        vbar += 1
                else:
                    if np.random.random() < d and v < 2 * states:
                        v += 1
                    if np.random.random() < b and vbar > 1:
                        vbar -= 1
                ta_state[f, 1] = v
                ta_state[f, 0] = vbar
    return ta_state


def generer_cible_et_table(n, rng):
    target = rng.integers(0, 3, size=n)
    if np.all(target == 2):
        target[0] = 1
    examples = np.array(list(itertools.product([0, 1], repeat=n)), dtype=np.int64)
    labels = np.ones(len(examples), dtype=np.int64)
    for idx, X in enumerate(examples):
        if any((x == 1 and t == 0) or (x == 0 and t == 1) for x, t in zip(X, target)):
            labels[idx] = 0
    t_signed = sorted(
        [i + 1 for i in range(n) if target[i] == 1] + [-i - 1 for i in range(n) if target[i] == 0]
    )
    return examples, labels, t_signed


def main():
    rng = np.random.default_rng(0)
    states, epochs, runs = 10000, 100, 100
    a, b, c, d = 1.0, 0.3, 1.0, 0.0
    p_pcl = 0.75
    s_tm, T_tm, clauses_tm = 1.1, 3, 1
    ns = list(range(4, 13))

    print(f"{'n':>3} {'TM-a':>8} {'PCL-a':>8} {'Notre regle-a':>14}   (sur {runs} runs)")
    for n in ns:
        succ_tm = succ_pcl = succ_notre = 0
        t0 = time.time()
        for _ in range(runs):
            examples, labels, t_signed = generer_cible_et_table(n, rng)

            # --- TM-a (code officiel tmu, memes reglages que TM_convergence_all.py) ---
            tm = TMClassifier(clauses_tm * 2, T_tm, s_tm, type_i_ii_ratio=1.0,
                               platform="CPU", boost_true_positive_feedback=0)
            Xu = examples.astype(np.uint32)
            Yu = labels.astype(np.uint32)
            for _ep in range(epochs):
                tm.fit(Xu, Yu)
            precision = tm.clause_precision(1, 0, Xu, Yu)
            recall = tm.clause_recall(1, 0, Xu, Yu)
            if np.all(precision == 1.0) and np.all(recall == 1.0):
                succ_tm += 1

            # --- PCL-a (code officiel, non modifie) ---
            result_pcl = PCL_run(n, epochs, examples, labels, 1, states, p_pcl)
            formula_pcl = sorted([j + 1 if result_pcl[0, j, 1] > states else -j - 1
                                   for j in range(n) if result_pcl[0, j, 0] > states or result_pcl[0, j, 1] > states])
            if formula_pcl == t_signed:
                succ_pcl += 1

            # --- Notre regle, meme protocole ---
            ta_state = notre_regle(n, epochs, examples, labels, states, a, b, c, d)
            formula_notre = sorted([j + 1 if ta_state[j, 1] > states else -j - 1
                                     for j in range(n) if ta_state[j, 0] > states or ta_state[j, 1] > states])
            if formula_notre == t_signed:
                succ_notre += 1

        dt = time.time() - t0
        print(f"{n:>3} {succ_tm:>5}/{runs} {succ_pcl:>5}/{runs} {succ_notre:>10}/{runs}   ({dt:.1f}s)")


if __name__ == "__main__":
    main()
