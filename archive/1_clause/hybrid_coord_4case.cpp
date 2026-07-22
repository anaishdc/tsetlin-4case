#include "tsetlin_engine_4case.h"
#include <fstream>
#include <sstream>

// Exemple labellise : ([1,0,1,0,0], 0)
struct LabeledExample {
    vector<int> x;
    int y;
};
#include <iostream>
#include <cmath>
#include <algorithm>

// -----------------------------------------------------------------
// Coordination de la vraie TM (plusieurs clauses +/- polarite,
// selection probabiliste via seuil T -- Eqn 8-11 du papier Granmo)
// MAIS la mise a jour reelle d'une clause selectionnee utilise NOTRE
// regle a 4 cas (Clause::update standard), pas Table 2/3 (Type I/II).
// Pas de routeur du tout.
// -----------------------------------------------------------------

vector<LabeledExample> loadDataset(const string& path, int n) {
    vector<LabeledExample> data;
    ifstream f(path); string line;
    while (getline(f, line)) {
        if (line.empty()) continue;
        istringstream iss(line);
        vector<int> x(n);
        for (int i = 0; i < n; i++) iss >> x[i];
        int y; iss >> y;
        data.push_back({x, y});
    }
    return data;
}

struct HybridTM {
    vector<Clause> clauses;
    vector<int> polarity; // +1 ou -1
    int T; double S, a;
    HybridTM(int n, int N, int C, int T_, double S_, double a_) : T(T_), S(S_), a(a_) {
        for (int k = 0; k < C; k++) { clauses.emplace_back(n, N); polarity.push_back(k % 2 == 0 ? +1 : -1); }
    }
    int rawSum(const vector<int>& x) const {
        int v = 0;
        for (size_t k = 0; k < clauses.size(); k++) v += polarity[k] * clauses[k].output(x);
        return v;
    }
    int output(const vector<int>& x) const { return rawSum(x) >= 0 ? 1 : 0; }
    double normalizedScore(const vector<int>& x) const {
        int v = max(-T, min(T, rawSum(x)));
        return (double)v / T;
    }
    void update(const vector<int>& x, int y) {
        int v = max(-T, min(T, rawSum(x)));
        Params p = {S, a};
        for (size_t k = 0; k < clauses.size(); k++) {
            bool pos = polarity[k] > 0;
            double prob = (y == 1) ? (T - v) / (2.0 * T) : (T + v) / (2.0 * T);
            if (!flip(prob)) continue;
            int localTarget = pos ? y : (1 - y);
            clauses[k].update(x, localTarget, p);
        }
    }
};

int main(int argc, char** argv) {
    string which = argc > 1 ? argv[1] : "xor";
    int n; vector<LabeledExample> trainRawAll, testRawAll;
    int N = 100, total = 30000, C = 20, T = 15, nbRuns = 10;
    double S = 1, a = 0.3;

    if (which == "xor") {
        n = 12;
        trainRawAll = loadDataset("../data/NoisyXORTrainingData.txt", n);
        testRawAll  = loadDataset("../data/NoisyXORTestData.txt", n);
    } else if (which == "mux11") {
        n = 11;
        trainRawAll = loadDataset("../data/Mux11TrainingData.txt", n);
        testRawAll  = loadDataset("../data/Mux11TestData.txt", n);
    } else { // parity3
        n = 12;
        trainRawAll = loadDataset("../data/Parity3TrainingData.txt", n);
        testRawAll  = loadDataset("../data/Parity3TestData.txt", n);
    }
    C = argc > 2 ? atoi(argv[2]) : 20;
    T = argc > 3 ? atoi(argv[3]) : 15;
    nbRuns = argc > 4 ? atoi(argv[4]) : 10;
    total = argc > 5 ? atoi(argv[5]) : 200000;
    if (argc > 6) S = atof(argv[6]);
    if (argc > 7) a = atof(argv[7]);
    if (argc > 8) N = atoi(argv[8]);

    cout << which << " (coordination vraie TM + notre regle a 4 cas, sans routeur) : C=" << C
         << " T=" << T << " S=" << S << " a=" << a << " N=" << N << "\n" << flush;

    vector<double> accs(nbRuns);
    for (int run = 0; run < nbRuns; run++) {
        rng.seed(120000 + run);
        HybridTM tm(n, N, C, T, S, a);
        for (int t = 0; t < total; t++) {
            const auto& ex = trainRawAll[rng() % trainRawAll.size()];
            tm.update(ex.x, ex.y);
        }
        int correct = 0;
        for (auto& ex : testRawAll) if (tm.output(ex.x) == ex.y) correct++;
        accs[run] = 100.0 * correct / testRawAll.size();

        long totalLit = 0;
        for (auto& c : tm.clauses) for (int i = 0; i < c.n; i++) { if (c.v[i].included()) totalLit++; if (c.vbar[i].included()) totalLit++; }
        cout << "  run " << run << " : acc=" << accs[run] << "%  complexite=" << totalLit << "\n" << flush;
    }
    double mean = 0; for (double a2 : accs) mean += a2; mean /= nbRuns;
    double var = 0; for (double a2 : accs) var += (a2-mean)*(a2-mean); var /= nbRuns;
    cout << which << " : Mean=" << mean << " +/- " << sqrt(var) << "\n";
    return 0;
}
