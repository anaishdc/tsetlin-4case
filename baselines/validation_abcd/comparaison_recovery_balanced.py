"""Reproduit le protocole "b" de la Table 7 de PCL (AAAI-25) : au lieu de
toute la table de verite a chaque epoch, on tire seulement 2 exemples par
epoch (1 positif + 1 negatif), sur 100 epochs. Reutilise leur code officiel
tel quel pour PCL-b (dpcl-classifier/Main_code_AAAI25/PCL_convergence_balanced.py)
et pour TM-b (meme fichier TM_convergence_balanced.py, bibliotheque tmu) ;
ajoute seulement notre regle avec exactement le meme protocole d'echantillonnage.

Usage: python3 comparaison_recovery_balanced.py
"""
import importlib.util
import itertools
import sys
import time

import numpy as np
from numba import njit

# --- PCL officiel, version "balanced" (protocole b), import direct du fichier ---
_spec = importlib.util.spec_from_file_location(
    "PCL_convergence_balanced", "../dpcl-classifier/Main_code_AAAI25/PCL_convergence_balanced.py")
_pclb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pclb)
PCL_balanced = _pclb.PCL  # PCL(n, ta_state, examples, labels, clauses, states, p) -- mutation en place

# --- TM officielle (tmu), memes reglages que TM_convergence_balanced.py ---
from tmu.models.classification.vanilla_classifier import TMClassifier  # noqa: E402


@njit(cache=True)
def notre_regle_balanced(n, ta_state, examples, labels, states, a, b, c, d):
    """Meme protocole que PCL_balanced : 2 exemples (examples/labels), etat
    ta_state modifie en place, appelee une fois par epoch depuis Python."""
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
        # cible degeneree (aucun litteral pertinent) -- meme garde que le code officiel PCL
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
    p_pcl = 0.7  # meme p que PCL-b dans leur Table 7
    s_tm, T_tm, clauses_tm = 1.1, 3, 1  # memes reglages que TM_convergence_balanced.py
    ns = list(range(4, 13))

    print(f"{'n':>3} {'TM-b':>8} {'PCL-b':>8} {'Notre regle-b':>14}   (sur {runs} runs)")
    for n in ns:
        succ_tm = succ_pcl = succ_notre = 0
        t0 = time.time()
        for _ in range(runs):
            examples, labels, t_signed = generer_cible_et_table(n, rng)
            X0 = examples[labels == 0]
            X1 = examples[labels == 1]
            Xb = np.zeros((2, n), dtype=np.uint32)
            Yb = np.array([1, 0], dtype=np.uint32)

            # --- TM-b (code officiel tmu, memes reglages) ---
            tm = TMClassifier(clauses_tm * 2, T_tm, s_tm, type_i_ii_ratio=1.0,
                               platform="CPU", boost_true_positive_feedback=0)
            for _ep in range(epochs):
                Xb[0] = X1[np.random.randint(X1.shape[0])]
                Xb[1] = X0[np.random.randint(X0.shape[0])]
                tm.fit(Xb, Yb)
            precision = tm.clause_precision(1, 0, examples.astype(np.uint32), labels.astype(np.uint32))
            recall = tm.clause_recall(1, 0, examples.astype(np.uint32), labels.astype(np.uint32))
            if np.all(precision == 1.0) and np.all(recall == 1.0):
                succ_tm += 1

            # --- PCL-b (code officiel, non modifie) ---
            ta_state_pcl = np.random.choice(np.array([states, states + 1]), size=(1, n, 2)).astype(np.int32)
            for _ep in range(epochs):
                Xb[0] = X1[np.random.randint(X1.shape[0])]
                Xb[1] = X0[np.random.randint(X0.shape[0])]
                PCL_balanced(n, ta_state_pcl, Xb.astype(np.int64), Yb.astype(np.int64), 1, states, p_pcl)
            formula_pcl = sorted([j + 1 if ta_state_pcl[0, j, 1] > states else -j - 1
                                   for j in range(n) if ta_state_pcl[0, j, 0] > states or ta_state_pcl[0, j, 1] > states])
            if formula_pcl == t_signed:
                succ_pcl += 1

            # --- Notre regle, meme protocole ---
            ta_state_notre = np.random.choice(np.array([states, states + 1]), size=(n, 2)).astype(np.int64)
            for _ep in range(epochs):
                Xb[0] = X1[np.random.randint(X1.shape[0])]
                Xb[1] = X0[np.random.randint(X0.shape[0])]
                ta_state_notre = notre_regle_balanced(n, ta_state_notre, Xb.astype(np.int64), Yb.astype(np.int64),
                                                        states, a, b, c, d)
            formula_notre = sorted([j + 1 if ta_state_notre[j, 1] > states else -j - 1
                                     for j in range(n) if ta_state_notre[j, 0] > states or ta_state_notre[j, 1] > states])
            if formula_notre == t_signed:
                succ_notre += 1

        dt = time.time() - t0
        print(f"{n:>3} {succ_tm:>5}/{runs} {succ_pcl:>5}/{runs} {succ_notre:>10}/{runs}   ({dt:.1f}s)")


if __name__ == "__main__":
    main()
