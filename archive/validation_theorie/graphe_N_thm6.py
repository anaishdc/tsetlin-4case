"""Validation directe et complete du Theoreme des 3 regimes (thm:sel) --
remplace l'ancienne figure (Test D, balayage en S a N fixe) qui ne testait
que le signe de la derive, pas la decroissance en N annoncee par le
theoreme.

Ici : loi P(X,Y) fixee (Delta>0, donc regime 1 : v doit finir Include,
vbar doit finir Exclude), on fait varier N, et on mesure la vraie quantite
du theoreme :
    limsup_t P((v_t, vbar_t) != (Inc, Exc))
comparee a la borne exacte gamma^{-N} + gamma_barre^{-N}.

Usage: python3 graphe_N_thm6.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numba import njit

rng = np.random.default_rng(3)

S, a = 1.0, 0.3
P00, P10, P01, P11 = 0.5, 0.2, 0.1, 0.2  # Delta>0 (regime 1)

p_plus = S * P00
p_minus = S * P10 + a * P01
pbar_plus = S * P10
pbar_minus = S * P00 + a * P11

Delta = p_plus - p_minus
Deltabar = pbar_plus - pbar_minus
rho = p_plus / p_minus
rhobar = pbar_plus / pbar_minus
gamma = max(rho, 1 / rho)
gammabar = max(rhobar, 1 / rhobar)

print(f"Delta={Delta:.4f} (attendu >0), Deltabar={Deltabar:.4f} (attendu <0)")
print(f"rho={rho:.4f} gamma={gamma:.4f} | rho_barre={rhobar:.4f} gamma_barre={gammabar:.4f}")


@njit(cache=True)
def simulate_joint_wrong_rate(S, a, P00, P10, P01, P11, N, T, burn, R):
    """R repetitions independantes de la VRAIE regle a 4 cas : un seul
    exemple (x,y) tire par pas de temps (partage par v et vbar, comme dans
    le vrai code -- update_masked du moteur C++), puis chaque automate
    concerne par ce cas fait son propre tirage Bernoulli(S ou a)
    independant pour savoir si son action se declenche. Retourne la
    fraction de temps (apres burn-in) ou (v,vbar) != (Inc,Exc)."""
    total_wrong = 0
    total_steps = 0
    for r in range(R):
        v = N
        vbar = N
        for t in range(T):
            u_case = np.random.random()
            if u_case < P00:
                # x=0,y=0 : v vers Include(S), vbar vers Exclude(S) -- tirages independants
                if np.random.random() < S:
                    if v < 2 * N:
                        v += 1
                if np.random.random() < S:
                    if vbar > 1:
                        vbar -= 1
            elif u_case < P00 + P10:
                # x=1,y=0 : v vers Exclude(S), vbar vers Include(S)
                if np.random.random() < S:
                    if v > 1:
                        v -= 1
                if np.random.random() < S:
                    if vbar < 2 * N:
                        vbar += 1
            elif u_case < P00 + P10 + P01:
                # x=0,y=1 : v vers Exclude(a), vbar : rien
                if np.random.random() < a:
                    if v > 1:
                        v -= 1
            else:
                # x=1,y=1 : v : rien, vbar vers Exclude(a)
                if np.random.random() < a:
                    if vbar > 1:
                        vbar -= 1

            if t >= burn:
                v_inc = v > N
                vbar_exc = vbar <= N
                if not (v_inc and vbar_exc):
                    total_wrong += 1
                total_steps += 1
    return total_wrong / total_steps


def main():
    Ns = [2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 24]
    T, burn = 15000, 3000
    useful = T - burn
    target_events = 40  # nb d'evenements "mauvais regime" vises par point

    # warm-up JIT
    simulate_joint_wrong_rate(S, a, P00, P10, P01, P11, 5, 100, 10, 2)

    emp = []
    Rs = []
    for N in Ns:
        bound = gamma ** (-N) + gammabar ** (-N)
        R = int(np.clip(target_events / (useful * bound), 200, 25_000))
        rate = simulate_joint_wrong_rate(S, a, P00, P10, P01, P11, N, T, burn, R)
        emp.append(rate)
        Rs.append(R)
        n_events = rate * useful * R
        print(f"N={N:3d}  R={R:7d}  empirique={rate:.3e}   borne={bound:.3e}   (~{n_events:.0f} evenements observes)")

    Ns = np.array(Ns)
    emp = np.array(emp)
    bound = gamma ** (-Ns.astype(float)) + gammabar ** (-Ns.astype(float))
    bound = np.minimum(bound, 1.0)

    nonzero = emp > 0
    if (~nonzero).any():
        print(f"(points a 0 evenement observe, non traces sur l'echelle log : "
              f"N={Ns[~nonzero].tolist()}, coherent avec une borne minuscule a ces N)")

    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    ax.plot(Ns[nonzero], emp[nonzero], "o-", color="#1b6ca8", linewidth=2, markersize=6,
            label=r"$P((v_t,\bar v_t)\neq(\mathrm{Inc},\mathrm{Exc}))$ empirique")
    ax.plot(Ns, bound, "s--", color="#c0392b", linewidth=2, markersize=5,
            label=r"borne $\gamma^{-N}+\bar\gamma^{-N}$ (Théorème des 3 régimes)")
    ax.set_yscale("log")
    ax.set_xlabel("N (nombre d'états par automate)")
    ax.set_ylabel("probabilité du mauvais régime (échelle log)")
    ax.set_title("Validation directe du Théorème des 3 régimes\n"
                  r"(paire $(v,\bar v)$, $\Delta>0$ fixé, borne $\gamma^{-N}+\bar\gamma^{-N}$)")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig("../figures/graphe_N_thm6.png", dpi=150)
    print("Figure ecrite dans figures/graphe_N_thm6.png")


if __name__ == "__main__":
    main()