
import itertools
import sys
import time

import numpy as np
from numba import njit

sys.path.insert(0, "../dpcl-classifier/Main_code_AAAI25")
from PCL_convergence_all import PCL as PCL_run  # noqa: E402

sys.path.insert(0, ".")
from pyTsetlinMachine.tm import MultiClassTsetlinMachine  # noqa: E402


# ---------------------------------------------------------------------------
# Notre regle generalisee a 4 parametres libres (a,b,c,d), meme protocole que
# PCL_run (epochs, ordre fixe, N etats). L'ancienne regle (S,alpha) est le cas
# particulier a=c=S, b=alpha, d=0.
# ---------------------------------------------------------------------------

@njit(cache=True)
def notre_regle(n, epochs, examples, labels, states, a, b, c, d):
    ta_state = np.random.choice(np.array([states, states + 1]), size=(n, 2)).astype(np.int64)
    for _ in range(epochs):
        for e in range(len(examples)):
            y = labels[e]
            for f in range(n):
                xf = examples[e][f]
                v = ta_state[f, 1]     # automate litteral positif x_f
                vbar = ta_state[f, 0]  # automate litteral negatif not x_f
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
        # cible degeneree (aucun litteral pertinent) -- meme garde que le code officiel PCL
        target[0] = 1
    examples = np.array(list(itertools.product([0, 1], repeat=n)), dtype=np.int64)
    labels = np.ones(len(examples), dtype=np.int64)
    for idx, X in enumerate(examples):
        if any((x == 1 and t == 0) or (x == 0 and t == 1) for x, t in zip(X, target)):
            labels[idx] = 0
    t_signed = []
    for i in range(n):
        if target[i] == 1:
            t_signed.append(i + 1)
        elif target[i] == 0:
            t_signed.append(-i - 1)
    return target, examples, labels, t_signed


def formule_notre_regle(ta_state, states, n):
    return [j + 1 if ta_state[j, 1] > states else -j - 1
            for j in range(n) if ta_state[j, 1] > states or ta_state[j, 0] > states]


def formule_tm(tm, n, class_idx=1):
    lits = []
    for j in range(n):
        pos = tm.ta_action(class_idx, 0, j) == 1       # litteral x_j (clause 0, positive polarity)
        neg = tm.ta_action(class_idx, 0, n + j) == 1    # litteral not x_j
        if pos:
            lits.append(j + 1)
        if neg:
            lits.append(-j - 1)
    return lits


def main():
    rng = np.random.default_rng(0)
    states, epochs, runs = 10000, 100, 100
    a, b, c, d = 1.0, 0.3, 1.0, 0.0  # a=c=S=1.0, b=alpha=0.3, d=0 (cas particulier valide, meme reglage qu'avant)
    ns = [4, 6, 8, 10, 12]

    print(f"{'n':>3} {'TM':>8} {'PCL':>8} {'Notre regle':>12}   (sur {runs} runs)")
    for n in ns:
        succ_tm = succ_pcl = succ_notre = 0
        t0 = time.time()
        for _ in range(runs):
            target, examples, labels, t_signed = generer_cible_et_table(n, rng)

            # --- PCL (code officiel, 1 clause) ---
            result_pcl = PCL_run(n, epochs, examples, labels, 1, states, 0.75)
            formulas_pcl = [[j + 1 if result_pcl[i, j, 1] > states else -j - 1
                             for j in range(n) if result_pcl[i, j, 0] > states or result_pcl[i, j, 1] > states]
                            for i in range(1)]
            if t_signed in formulas_pcl:
                succ_pcl += 1

            # --- Notre regle (1 clause) ---
            ta_state = notre_regle(n, epochs, examples, labels, states, a, b, c, d)
            formula_notre = formule_notre_regle(ta_state, states, n)
            if formula_notre == t_signed:
                succ_notre += 1

            # --- TM officielle (pyTsetlinMachine, 2 clauses = 1 paire +/-) ---
            tm = MultiClassTsetlinMachine(2, 15, 3.9)
            tm.fit(examples.astype(np.int32), labels.astype(np.int32), epochs=epochs)
            formula_tm = formule_tm(tm, n)
            if sorted(formula_tm) == sorted(t_signed):
                succ_tm += 1

        dt = time.time() - t0
        print(f"{n:>3} {succ_tm:>5}/{runs} {succ_pcl:>5}/{runs} {succ_notre:>8}/{runs}   ({dt:.1f}s)")


if __name__ == "__main__":
    main()
