"""Graphe : effet de N sur l'erreur empirique P(C_t != C*) vs la borne du
Theoreme 7 (concentration multi-bit) : 2(n-K) * gamma_min^{-N}.

Simulation fidele a la vraie regle : un seul exemple (x1,x2,x3,y) est tire
par pas de temps -- loi jointe construite par independance conditionnelle
a Y (P(x1,x2,x3,y) = P(y) * P(x1|y) * P(x2|y) * P(x3|y), une vraie loi
jointe sur 3 features "correlees via Y", pas 3 scenarios independants
bricoles separement). Chacun des 6 automates (v_j, vbar_j, j=1,2,3) est
mis a jour par la VRAIE regle a 4 cas a partir de ce tirage partage.

Usage: python3 graphe_N_thm7.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numba import njit

S, a = 1.0, 0.3
p_y1 = 0.5  # P(Y=1)

# Pour chaque feature j : P(Xj=1|Y=0)=q0_j, P(Xj=1|Y=1)=q1_j
# feature 1 : tres separee (facile) ; feature 2 : moyenne ; feature 3 : peu separee (maillon faible)
q0 = np.array([0.10, 0.35, 0.40])
q1 = np.array([0.90, 0.70, 0.62])
n_feat = 3


def deltas_rho(q0_j, q1_j, p_y1, S, a):
    P00 = (1 - p_y1) * (1 - q0_j)
    P10 = (1 - p_y1) * q0_j
    P01 = p_y1 * (1 - q1_j)
    P11 = p_y1 * q1_j
    p_plus = S * P00
    p_minus = S * P10 + a * P01
    pbar_plus = S * P10
    pbar_minus = S * P00 + a * P11
    rho = p_plus / p_minus
    rhobar = pbar_plus / pbar_minus
    return rho, rhobar


gammas = []
rho_gt_1 = np.zeros(n_feat, dtype=np.bool_)
for j in range(n_feat):
    rho, rhobar = deltas_rho(q0[j], q1[j], p_y1, S, a)
    gammas.append(max(rho, 1 / rho))
    gammas.append(max(rhobar, 1 / rhobar))
    rho_gt_1[j] = rho > 1
    print(f"feature {j+1}: rho={rho:.4f} (gamma={gammas[-2]:.3f})  "
          f"rho_barre={rhobar:.4f} (gamma_barre={gammas[-1]:.3f})  "
          f"regime attendu = {'(Inc,Exc)' if rho>1 else '(Exc,Inc)'}")
gamma_min = min(gammas)
d = n_feat
print(f"gamma_min = {gamma_min:.4f}  (2d={2*d} automates)")
assert rho_gt_1.all(), "toutes les features doivent avoir rho>1 pour la formule de verification simplifiee ci-dessous"


@njit(cache=True)
def simulate_wrong_rate(q0, q1, p_y1, S, a, N, T, burn, R):
    """R repetitions. A chaque pas : tire y~Bernoulli(p_y1) puis, pour
    chaque feature j, x_j~Bernoulli(q1[j]) si y=1 sinon Bernoulli(q0[j])
    -- un seul (x1,x2,x3,y) partage -- puis met a jour les 6 automates
    par la regle a 4 cas. Ici rho_j>1 pour tous les j (verifie avant
    l'appel), donc l'etat correct est (Inc,Exc) pour chaque paire."""
    total_wrong = 0
    total_steps = 0
    for r in range(R):
        state = np.empty((n_feat, 2), dtype=np.int64)
        for f in range(n_feat):
            state[f, 0] = N
            state[f, 1] = N + 1
        for t in range(T):
            y = 1 if np.random.random() < p_y1 else 0
            for f in range(n_feat):
                qf = q1[f] if y == 1 else q0[f]
                xf = 1 if np.random.random() < qf else 0
                v = state[f, 1]
                vbar = state[f, 0]
                if xf == 0:
                    if y == 0:
                        if np.random.random() < S and v < 2 * N:
                            v += 1
                        if np.random.random() < S and vbar > 1:
                            vbar -= 1
                    else:
                        if np.random.random() < a and v > 1:
                            v -= 1
                else:
                    if y == 0:
                        if np.random.random() < S and v > 1:
                            v -= 1
                        if np.random.random() < S and vbar < 2 * N:
                            vbar += 1
                    else:
                        if np.random.random() < a and vbar > 1:
                            vbar -= 1
                state[f, 1] = v
                state[f, 0] = vbar

            if t >= burn:
                any_wrong = False
                for f in range(n_feat):
                    v_inc = state[f, 1] > N
                    vbar_exc = state[f, 0] <= N
                    if not (v_inc and vbar_exc):
                        any_wrong = True
                        break
                if any_wrong:
                    total_wrong += 1
                total_steps += 1
    return total_wrong / total_steps


def main():
    n_feat_local = n_feat
    Ns = [2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 26, 32, 40]
    T, burn = 20000, 3000
    useful = T - burn
    target_events = 40

    # warm-up JIT
    simulate_wrong_rate(q0, q1, p_y1, S, a, 5, 100, 10, 2)

    emp = []
    for N in Ns:
        bound = min(1.0, 2 * d * gamma_min ** (-float(N)))
        R = int(np.clip(target_events / (useful * bound), 200, 25_000))
        rate = simulate_wrong_rate(q0, q1, p_y1, S, a, N, T, burn, R)
        emp.append(rate)
        n_events = rate * useful * R
        print(f"N={N:3d}  R={R:6d}  empirique={rate:.3e}  borne={bound:.3e}  (~{n_events:.0f} evenements)")

    Ns = np.array(Ns)
    emp = np.array(emp)
    bound = np.minimum(1.0, 2 * d * gamma_min ** (-Ns.astype(float)))

    nonzero = emp > 0
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(Ns[nonzero], emp[nonzero], "o-", color="#1b6ca8",
            label=r"$P(C_t \neq C^\star)$ empirique", linewidth=2, markersize=6)
    ax.plot(Ns, bound, "s--", color="#c0392b",
            label=r"borne $2(n{-}K)\,\gamma_{\min}^{-N}$ (Théorème 7)", linewidth=2, markersize=5)
    ax.set_yscale("log")
    ax.set_xlabel(r"$N$ (nombre d'états par automate)")
    ax.set_ylabel(r"$P(C_t \neq C^\star)$ (échelle log)")
    ax.set_title("Effet de $N$ sur l'erreur de récupération structurelle\n(validation directe du Théorème 7, simulation fidèle)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig("../figures/graphe_N_thm7.png", dpi=150)
    print("Figure ecrite dans figures/graphe_N_thm7.png")


if __name__ == "__main__":
    main()
