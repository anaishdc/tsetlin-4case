#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string> 
#include <algorithm>
#include <cmath>
#include <random>
#include <functional>

using namespace std;

// Fonction cible : prend un vecteur d'entrée, retourne 0 ou 1
using Target  = function<int(const vector<int>&)>;

// Jeu de test : liste de paires (entrée x, vrai y)
using TestSet = vector<pair<vector<int>, int>>;

// Générateur aléatoire global -- thread_local : une instance independante
// par thread (au lieu d'un seul mt19937 partage) pour permettre la
// parallelisation (ex. un thread par classe) sans race condition sur
// l'etat interne du generateur. Ne change rien au comportement mono-thread
// existant (un seul thread => un seul rng, identique a avant).
static thread_local mt19937 rng(42);
static thread_local uniform_real_distribution<double> uniform01(0.0, 1.0);
static bool flip(double p) { return uniform01(rng) < p; }

// Convertit un index entier en vecteur binaire de taille n
// ex : n=3, idx=5 (101) -> {1, 0, 1}
inline vector<int> fromIndex(int idx, int n) {
    vector<int> x(n);
    for (int i = 0; i < n; i++) x[i] = (idx >> (n - 1 - i)) & 1;
    return x;
}

// ---------------------------------------------------------------
// Structure logique cible :
// ---------------------------------------------------------------
struct LiteralSet {
    vector<bool> pos;  // pos[i] = vrai si xi  doit être inclus
    vector<bool> neg;  // neg[i] = vrai si ¬xi doit être inclus
};

// ---------------------------------------------------------------
// Paramètres de la règle d'apprentissage à 4 cas : seulement S et a
// (S1 et a1 sont réutilisés comme S et a, le reste n'est pas utilisé)
// ---------------------------------------------------------------
struct Params {
    double S1, S2, S3, S4, a1, a2;
};

// ---------------------------------------------------------------
// Automate de Tsetlin
// ---------------------------------------------------------------
struct Automaton {
    int state, N;
    Automaton(int N_, int initState) : state(initState), N(N_) {}
    int  included()              const { return state > N ? 1 : 0; }
    void towardInclude(double p) { if (flip(p)) state = min(state + 1, 2 * N); }
    void towardExclude(double p) { if (flip(p)) state = max(state - 1, 1);     }
};

// ---------------------------------------------------------------
// Clause logique
// ---------------------------------------------------------------
struct Clause {
    int n;
    vector<Automaton> v, vbar;

    // Init symetrique (Florent) : pour chaque feature i, v[i]=s et vbar[i]=2N+1-s
    // => exactement l'un des deux est inclus, l'autre exclu. Pas de xi ^ ¬xi au depart.
    Clause(int n_, int N) : n(n_) {
        uniform_int_distribution<int> st(1, 2 * N);
        for (int i = 0; i < n; i++) {
            int s = st(rng);
            v.emplace_back(N, s);
            vbar.emplace_back(N, 2 * N + 1 - s);
        }
    }

    // Retourne 1 si aucun littéral inclus n'est violé par x
    int output(const vector<int>& x) const {
        for (int i = 0; i < n; i++) {
            if (v[i].included()    && x[i] == 0) return 0;
            if (vbar[i].included() && x[i] == 1) return 0;
        }
        return 1;
    }

    // Règle d'apprentissage à 4 cas (table (x_i,y) -> action), appliquée
    // symétriquement à v[i] (littéral x_i) et vbar[i] (littéral ¬x_i) :
    //
    //   x_i | y | v[i] | vbar[i]
    //   ----+---+------+--------
    //    0  | 0 |  +S  |   -S
    //    0  | 1 |  -a  |    0
    //    1  | 0 |  -S  |   +S
    //    1  | 1 |   0  |   -a
    //
    void update(const vector<int>& x, int y, const Params& p) {
        double S = p.S1, a = p.a1;
        for (int i = 0; i < n; i++) {
            if (x[i] == 0) {
                if (y == 0) { v[i].towardInclude(S); vbar[i].towardExclude(S); }
                else        { v[i].towardExclude(a); }
            } else {
                if (y == 0) { v[i].towardExclude(S); vbar[i].towardInclude(S); }
                else        { vbar[i].towardExclude(a); }
            }
        }
    }

    // Vrai si tous les automates sont exactement dans le bon état (convergence structurelle)
    bool isStructurallyExact(const LiteralSet& ls) const {
        for (int i = 0; i < n; i++) {
            if ((v[i].included()    == 1) != ls.pos[i]) return false;
            if ((vbar[i].included() == 1) != ls.neg[i]) return false;
        }
        return true;
    }

    // % de prédictions correctes sur un jeu de test propre
    double accuracy(const TestSet& testSet) const {
        int correct = 0;
        for (auto& [x, y] : testSet)
            if (output(x) == y) correct++;
        return correct / (double)testSet.size();
    }

    // Représentation textuelle de la clause courante
    string toString() const {
        vector<string> lits;
        for (int i = 0; i < n; i++) {
            if (v[i].included())    lits.push_back("x" + to_string(i + 1));
            if (vbar[i].included()) lits.push_back("¬x" + to_string(i + 1));
        }
        if (lits.empty()) return "(vide)";
        string s = lits[0];
        for (size_t k = 1; k < lits.size(); k++) s += " ^ " + lits[k];
        return s;
    }
};

// ---------------------------------------------------------------
// Métriques  : moyenne + écart-type pour exact et accuracy
// ---------------------------------------------------------------
struct Metrics {
    double exactRate, accuracy; // moyenne
    double stdExactRate = 0, stdAccuracy = 0; // écart-type
};


// Génère les 2^n entrées possibles avec leurs vrais labels (sans bruit)
inline TestSet makeTestSet(int n, const Target& target) {
    TestSet ts;
    for (int idx = 0; idx < (1 << n); idx++) {
        auto x = fromIndex(idx, n);
        ts.push_back({x, target(x)});
    }
    return ts;
}


// Simule un seul essai d'apprentissage, évalue sur la fenêtre finale
inline Metrics runTrial(unsigned seed, int n, const Target& target,
                        const LiteralSet& ls, const TestSet& testSet,
                        int totalExamples, int window, double noiseProb,
                        const Params& p, int N = 2) {
    rng.seed(seed);
    Clause clause(n, N);
    uniform_int_distribution<int> bit(0, 1);
    int    onTarget    = 0;
    double sumAccuracy = 0.0;
    for (int t = 1; t <= totalExamples; t++) {
        vector<int> x(n);
        for (int& xi : x) xi = bit(rng);
        int yTrue  = target(x);
        int yNoisy = flip(noiseProb) ? (1 - yTrue) : yTrue;
        clause.update(x, yNoisy, p);
        if (t > totalExamples - window) {
            if (clause.isStructurallyExact(ls)) onTarget++;
            sumAccuracy += clause.accuracy(testSet);
        }
    }
    return { onTarget / (double)window, sumAccuracy / window };
}

// Lance un essai et retourne la clause apprise sous forme de texte (ex: "x1 ^ ~x2")
inline string learnedClauseStr(unsigned seed, int n, const Target& target,
                               int totalExamples, double noiseProb,
                               const Params& p, int N = 2) {
    rng.seed(seed);
    Clause clause(n, N);
    uniform_int_distribution<int> bit(0, 1); // générer au hasard des 0 ou des 1
    for (int t = 1; t <= totalExamples; t++) {
        vector<int> x(n);
        for (int& xi : x) xi = bit(rng);
        int yTrue  = target(x);
        int yNoisy = flip(noiseProb) ? (1 - yTrue) : yTrue;
        clause.update(x, yNoisy, p);
    }
    return clause.toString();
}

// Lance nbRuns essais indépendants, retourne moyenne + écart-type
inline Metrics runBatch(double noiseProb, int n, const Target& target,
                        const LiteralSet& ls, const TestSet& testSet,
                        int totalExamples, int window, int nbRuns,
                        const Params& p, int N = 2) {
    vector<double> ots(nbRuns), accs(nbRuns);
    for (int run = 0; run < nbRuns; run++) {
        Metrics m = runTrial(1000 + run, n, target, ls, testSet,
                             totalExamples, window, noiseProb, p, N);
        ots[run]  = m.exactRate;
        accs[run] = m.accuracy;
    }
    double meanOT = 0, meanAcc = 0;
    for (int i = 0; i < nbRuns; i++) { meanOT += ots[i]; meanAcc += accs[i]; }
    meanOT /= nbRuns; meanAcc /= nbRuns;
    double varOT = 0, varAcc = 0;
    for (int i = 0; i < nbRuns; i++) {
        varOT  += (ots[i]  - meanOT)  * (ots[i]  - meanOT);
        varAcc += (accs[i] - meanAcc) * (accs[i] - meanAcc);
    }
    return { meanOT, meanAcc, sqrt(varOT / nbRuns), sqrt(varAcc / nbRuns) };
}

// Écrit un tableau de résultats dans un fichier CSV
inline void writeCsv(const string& path, const string& colX,
                     const vector<pair<double, Metrics>>& rows) {
    ofstream f(path);
    f << colX << ",temps_sur_C,std_exact,accuracy,std_accuracy\n";
    for (auto& [x, m] : rows)
        f << x << "," << m.exactRate << "," << m.stdExactRate
          << "," << m.accuracy << "," << m.stdAccuracy << "\n";
    cout << "-> " << path << "\n";
}
