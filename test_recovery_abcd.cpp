// Reproduit le protocole officiel de PCL (DPCL_ConvergenceTest.py, AAAI-25) :
// cible aleatoire (chaque feature 0/1/inutile, tiree uniformement, avec garde
// contre la cible entierement inutile), table de verite complete, 100 epochs,
// 100 runs, succes = litteraux appris == litteraux cibles exactement.
// Utilise regle_abcd.h (notre regle generalisee a,b,c,d), teste ici avec
// a=c=1.0, b=0.3, d=0.0 (cas particulier deja valide, meme reglage que les
// comparaisons Python).
//
// Compilation : g++ -O2 -std=c++17 -I. test_recovery_abcd.cpp -o test_recovery_abcd
// Usage       : ./test_recovery_abcd

#include "regle_abcd.h"
#include <iostream>

int main() {
    const Params p = {1.0, 0.3, 1.0, 0.0}; // a, b, c, d
    const int N = 100;      // meme "states" que le protocole officiel PCL
    const int epochs = 100;
    const int runs = 100;
    const vector<int> ns = {4, 5, 6, 7, 8, 9, 10, 11, 12};

    cout << "Notre regle (a,b,c,d), protocole PCL officiel (DPCL_ConvergenceTest.py)\n";
    cout << "a=" << p.a << " b=" << p.b << " c=" << p.c << " d=" << p.d
         << "  N=" << N << " epochs=" << epochs << " runs=" << runs << "\n\n";
    cout << " n   succes\n";

    mt19937 gen(0);
    uniform_int_distribution<int> tri(0, 2); // 0=neg, 1=pos, 2=inutile

    for (int n : ns) {
        int succ = 0;
        for (int run = 0; run < runs; run++) {
            // --- genere la cible (meme schema que np.random.choice(3, n)) ---
            vector<int> target(n);
            bool allIrrelevant = true;
            for (int i = 0; i < n; i++) {
                target[i] = tri(gen);
                if (target[i] != 2) allIrrelevant = false;
            }
            if (allIrrelevant) target[0] = 1; // meme garde que le code officiel

            LiteralSet ls;
            ls.pos.assign(n, false);
            ls.neg.assign(n, false);
            for (int i = 0; i < n; i++) {
                if (target[i] == 1) ls.pos[i] = true;
                if (target[i] == 0) ls.neg[i] = true;
            }
            Target tgt = [&target](const vector<int>& x) {
                for (size_t i = 0; i < target.size(); i++) {
                    if (target[i] == 1 && x[i] == 0) return 0;
                    if (target[i] == 0 && x[i] == 1) return 0;
                }
                return 1;
            };

            // --- entraine une clause sur toute la table de verite, 100 epochs ---
            Clause clause(n, N);
            for (int ep = 0; ep < epochs; ep++) {
                for (int idx = 0; idx < (1 << n); idx++) {
                    vector<int> x = fromIndex(idx, n);
                    clause.update(x, tgt(x), p);
                }
            }
            if (clause.isStructurallyExact(ls)) succ++;
        }
        cout << " " << n << "   " << succ << "/" << runs << "\n";
    }
    return 0;
}
