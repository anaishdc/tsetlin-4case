"""Validation experimentale des theoremes de theorie_clean.tex.

Simule directement la dynamique d'un automate (ou d'une paire, ou d'une
conjonction de m littéraux) selon la regle a 4 cas -- independamment du
classifieur complet (pas de gates, pas de boosting, pas de dataset reel).

5 experiences :
  1. Loi stationnaire empirique vs formule exacte (Theoreme 1)
  2. Probabilite du mauvais cote vs N, echelle log (Theoreme 1 / Selection)
  3. Convergence de la moyenne temporelle vers pi(Inc) (Theoreme 1)
  4. Les 3 regimes + le cas frontiere Delta=0 (Theoreme Selection)
  5. Seuil de bruit eta* = 2S/(4S+a*2^m) sur une conjonction (Theoreme Identification)

Usage: python3 valider_theorie.py
"""
import numpy as np
from numba import njit

rng = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# Simulation d'un automate isole selon la regle a 4 cas (JIT numba)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _simulate_core(p_plus, p_minus, N, T, state0, u):
    state = state0
    history = np.empty(T, dtype=np.int64)
    for t in range(T):
        r = u[t]
        if r < p_plus:
            if state < 2 * N:
                state += 1
        elif r < p_plus + p_minus:
            if state > 1:
                state -= 1
        history[t] = state
    return history


def simulate_automaton(p_plus, p_minus, N, T, state0=None):
    """Simule un automate a 2N etats. Retourne l'historique des etats (1..2N)."""
    s0 = state0 if state0 is not None else N
    u = rng.random(T)
    return _simulate_core(p_plus, p_minus, N, T, s0, u)


def exact_pi(N, rho):
    k = np.arange(1, 2 * N + 1)
    if abs(rho - 1.0) < 1e-12:
        return np.full(2 * N, 1.0 / (2 * N))
    pi1 = (1 - rho) / (1 - rho ** (2 * N))
    return pi1 * rho ** (k - 1)


def pi_inc_exact(N, rho):
    return rho ** N / (1 + rho ** N)


# ---------------------------------------------------------------------------
# Experience 1 : loi stationnaire empirique vs exacte
# ---------------------------------------------------------------------------

def experience_1():
    print("=== Experience 1 : loi stationnaire pi(k) empirique vs exacte ===")
    S, a = 1.0, 0.3
    P00, P10, P01 = 0.5, 0.2, 0.1  # loi P(X,Y) arbitraire fixee
    p_plus = S * P00
    p_minus = S * P10 + a * P01
    rho = p_plus / p_minus
    N = 20
    T = 2_000_000
    burn = 100_000

    hist = simulate_automaton(p_plus, p_minus, N, T)
    hist = hist[burn:]
    pi_hat = np.bincount(hist, minlength=2 * N + 1)[1:] / len(hist)
    pi_exact = exact_pi(N, rho)

    err = np.max(np.abs(pi_hat - pi_exact))
    print(f"  rho={rho:.4f}, N={N}, T={T} (apres burn-in {burn})")
    print(f"  ecart max |pi_hat - pi_exact| sur les {2*N} etats = {err:.5f}")
    print(f"  pi(Inc) empirique = {pi_hat[N:].sum():.5f}, exact = {pi_inc_exact(N, rho):.5f}")
    print(f"  -> {'OK' if err < 0.01 else 'ECART TROP GRAND'}\n")


# ---------------------------------------------------------------------------
# Experience 2 : probabilite du mauvais cote vs N (echelle log)
# ---------------------------------------------------------------------------

def experience_2():
    print("=== Experience 2 : P(mauvais cote) vs N, decroissance en gamma^-N ===")
    S, a = 1.0, 0.3
    P00, P10, P01 = 0.5, 0.2, 0.1
    p_plus = S * P00
    p_minus = S * P10 + a * P01
    rho = p_plus / p_minus
    gamma = max(rho, 1 / rho)

    T = 200_000
    n_reps = 200
    print(f"  rho={rho:.4f}, gamma={gamma:.4f}")
    print(f"  {'N':>4} {'P(mauvais cote) empirique':>28} {'borne 1/(1+gamma^N)':>22} {'log10(empirique)':>18}")
    for N in [2, 4, 6, 8, 10, 14]:
        wrong = 0
        for _ in range(n_reps):
            hist = simulate_automaton(p_plus, p_minus, N, T)
            final_state = hist[-1]
            is_inc = final_state > N
            wrong += 0 if is_inc else 1  # rho>1 donc Include est le bon cote
        p_wrong_emp = wrong / n_reps
        p_wrong_th = 1.0 / (1 + gamma ** N)
        logval = np.log10(p_wrong_emp) if p_wrong_emp > 0 else float("-inf")
        print(f"  {N:>4} {p_wrong_emp:>28.5f} {p_wrong_th:>22.5f} {logval:>18.3f}")
    print("  -> la decroissance doit etre approximativement lineaire en N (log10)\n")


# ---------------------------------------------------------------------------
# Experience 3 : moyenne temporelle -> pi(Inc)
# ---------------------------------------------------------------------------

def experience_3():
    print("=== Experience 3 : moyenne temporelle 1/T sum 1[v_t>N] -> pi(Inc) ===")
    S, a = 1.0, 0.3
    P00, P10, P01 = 0.5, 0.2, 0.1
    p_plus = S * P00
    p_minus = S * P10 + a * P01
    rho = p_plus / p_minus
    N = 15
    T = 3_000_000

    hist = simulate_automaton(p_plus, p_minus, N, T)
    running_freq = np.cumsum(hist > N) / np.arange(1, T + 1)
    final_freq = running_freq[-1]
    exact = pi_inc_exact(N, rho)
    print(f"  N={N}, T={T}, rho={rho:.4f}")
    print(f"  frequence temporelle finale = {final_freq:.5f}, pi(Inc) exact = {exact:.5f}")
    print(f"  ecart = {abs(final_freq - exact):.5f}")
    print(f"  -> {'OK' if abs(final_freq - exact) < 0.01 else 'ECART TROP GRAND'}\n")


# ---------------------------------------------------------------------------
# Experience 4 : les 3 regimes + le cas frontiere
# ---------------------------------------------------------------------------

def deltas(S, a, P00, P10, P01, P11):
    Delta = S * P00 - S * P10 - a * P01
    Deltabar = S * P10 - S * P00 - a * P11
    p_plus, p_minus = S * P00, S * P10 + a * P01
    pbar_plus, pbar_minus = S * P10, S * P00 + a * P11
    return Delta, Deltabar, (p_plus, p_minus), (pbar_plus, pbar_minus)


def repeated_frac_include(p_plus, p_minus, N, T, burn, R):
    """R repetitions independantes, burn-in, retourne (moyenne, IC95%) de la
    fraction de temps passe en Include apres burn-in."""
    fracs = np.empty(R)
    for r in range(R):
        hist = simulate_automaton(p_plus, p_minus, N, T, state0=N)
        fracs[r] = np.mean(hist[burn:] > N)
    mean = fracs.mean()
    se = fracs.std(ddof=1) / np.sqrt(R)
    ci95 = 1.96 * se
    return mean, ci95


def experience_4():
    print("=== Experience 4 : les 3 regimes + le cas frontiere Delta=0 ===")
    print("    (R=30 repetitions independantes, burn-in, moyenne +/- IC95%)")
    S, a = 1.0, 0.3
    N, T, burn, R = 15, 300_000, 50_000, 30

    cases = {
        "Delta>0 (IDENTITY attendu)": (0.5, 0.1, 0.05, 0.35),
        "Deltabar>0 (NOT attendu)": (0.1, 0.5, 0.35, 0.05),
        "Delta<0 et Deltabar<0 (neutre attendu)": (0.3, 0.3, 0.2, 0.2),
        "Delta=0 (frontiere, aucune preference attendue)": None,
    }

    for name, probs in cases.items():
        if probs is None:
            # construire P00,P10,P01 pour forcer Delta = S*P00-S*P10-a*P01 = 0 exactement
            P00, P01, P11 = 0.3, 0.2, 0.2
            P10 = P00 - a * P01 / S
        else:
            P00, P10, P01, P11 = probs
        Delta, Deltabar, (pp, pm), (pbp, pbm) = deltas(S, a, P00, P10, P01, P11)

        mean_v, ci_v = repeated_frac_include(pp, pm, N, T, burn, R)
        mean_vbar, ci_vbar = repeated_frac_include(pbp, pbm, N, T, burn, R)

        print(f"  {name}")
        print(f"    Delta={Delta:.4f} Deltabar={Deltabar:.4f}")
        print(f"    P(v en Include) = {mean_v:.4f} +/- {ci_v:.4f} (IC95%)  "
              f"P(vbar en Include) = {mean_vbar:.4f} +/- {ci_vbar:.4f} (IC95%)")
    print()


# ---------------------------------------------------------------------------
# Experience 5 : seuil de bruit sur une conjonction (Theoreme identification)
# ---------------------------------------------------------------------------

def experience_5():
    print("=== Experience 5 : seuil de bruit eta* = 2S/(4S+a*2^m) ===")
    S, a, m = 1.0, 0.3, 3
    eta_star = 2 * S / (4 * S + a * 2 ** m)
    print(f"  S={S}, a={a}, m={m} -> seuil theorique eta* = {eta_star:.4f}")

    n = 5
    Jplus = list(range(m))
    N_samples = 2_000_000

    print(f"  {'eta':>8} {'Delta_j empirique':>20} {'Delta_j formule':>18} {'selectionne?':>14}")
    for eta in [eta_star - 0.1, eta_star - 0.02, eta_star, eta_star + 0.02, eta_star + 0.1]:
        eta = max(0.001, min(0.499, eta))
        X = rng.integers(0, 2, size=(N_samples, n))
        Ystar = np.all(X[:, Jplus] == 1, axis=1).astype(int)
        eps = rng.binomial(1, eta, size=N_samples)
        Y = Ystar ^ eps

        xj = X[:, 0]
        def prob(a_, b_):
            return np.mean((xj == a_) & (Y == b_))

        P00, P10, P01 = prob(0, 0), prob(1, 0), prob(0, 1)
        Delta_emp = S * P00 - S * P10 - a * P01
        Delta_formula = S * (1 - 2 * eta) * 2 ** (-m) - a * eta / 2
        print(f"  {eta:>8.4f} {Delta_emp:>20.5f} {Delta_formula:>18.5f} "
              f"{'oui' if Delta_emp > 0 else 'non':>14}")
    print()


if __name__ == "__main__":
    experience_1()
    experience_2()
    experience_3()
    experience_4()
    experience_5()
