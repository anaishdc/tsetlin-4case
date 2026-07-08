#include "../tsetlin_engine.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>

// Comme random_stump_perclause_boost.cpp, mais chaque clause fixe
// PLUSIEURS features aleatoires (pas juste 1) -- pour les problemes
// ayant besoin de plusieurs conditions simultanees (Parity, Mux).

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

struct RandomStumpClause {
    Clause clause;
    vector<int> condFeatures, condValues;
    vector<bool> active;
    RandomStumpClause(int n, int N, const vector<int>& feats, const vector<int>& vals)
        : clause(n, N), condFeatures(feats), condValues(vals), active(n, true) {
        for (size_t k = 0; k < feats.size(); k++) {
            clause.v[feats[k]].state = 1; clause.vbar[feats[k]].state = 1;
            active[feats[k]] = false;
        }
    }
    bool applies(const vector<int>& x) const {
        for (size_t k = 0; k < condFeatures.size(); k++) if (x[condFeatures[k]] != condValues[k]) return false;
        return true;
    }
    bool isEmpty() const {
        for (int i = 0; i < clause.n; i++) if (clause.v[i].included() || clause.vbar[i].included()) return false;
        return true;
    }
};

int main(int argc, char** argv) {
    string which = argc > 1 ? argv[1] : "parity3";
    int n; vector<LabeledExample> trainRawAll, testRawAll;
    int N = 100, total = 20000, M = 100, nbRuns = 5, K = 2;
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
    M = argc > 2 ? atoi(argv[2]) : 100;
    K = argc > 3 ? atoi(argv[3]) : 2;
    nbRuns = argc > 4 ? atoi(argv[4]) : 5;
    total = argc > 5 ? atoi(argv[5]) : 20000;
    if (argc > 6) S = atof(argv[6]);
    if (argc > 7) a = atof(argv[7]);
    if (argc > 8) N = atoi(argv[8]);

    cout << which << " (stumps a " << K << " features aleatoires, notre regle, sans routeur) : M=" << M
         << " total=" << total << " S=" << S << " a=" << a << " N=" << N << "\n" << flush;

    Params p = {S, S, S, S, a, a};
    uniform_int_distribution<int> valDist(0, 1);

    vector<double> accs(nbRuns);
    for (int run = 0; run < nbRuns; run++) {
        rng.seed(160000 + run);
        int Nex = (int)trainRawAll.size();
        vector<double> w(Nex, 1.0 / Nex);
        vector<RandomStumpClause> clauses;
        vector<double> alphas;

        for (int m = 0; m < M; m++) {
            vector<int> feats, vals;
            vector<int> pool(n); for (int i = 0; i < n; i++) pool[i] = i;
            shuffle(pool.begin(), pool.end(), rng);
            for (int k = 0; k < K; k++) { feats.push_back(pool[k]); vals.push_back(valDist(rng)); }
            clauses.emplace_back(n, N, feats, vals);

            vector<int> subsetIdx;
            for (int i = 0; i < Nex; i++) if (clauses.back().applies(trainRawAll[i].x)) subsetIdx.push_back(i);
            if (subsetIdx.empty()) { alphas.push_back(0.0); continue; }

            vector<int> subPos, subNeg;
            for (int i : subsetIdx) (trainRawAll[i].y == 1 ? subPos : subNeg).push_back(i);
            vector<double> wPos, wNeg;
            for (int i : subPos) wPos.push_back(w[i]);
            for (int i : subNeg) wNeg.push_back(w[i]);
            discrete_distribution<int> posDist, negDist;
            if (!wPos.empty()) posDist = discrete_distribution<int>(wPos.begin(), wPos.end());
            if (!wNeg.empty()) negDist = discrete_distribution<int>(wNeg.begin(), wNeg.end());

            for (int t = 0; t < total; t++) {
                bool useNeg = subNeg.empty() ? false : (subPos.empty() ? true : !flip(0.5));
                int idx = useNeg ? subNeg[negDist(rng)] : subPos[posDist(rng)];
                updateMasked(clauses.back().clause, trainRawAll[idx].x, trainRawAll[idx].y, p, clauses.back().active);
            }

            double err = 0, wSum = 0;
            vector<bool> wrong(subsetIdx.size());
            for (size_t j = 0; j < subsetIdx.size(); j++) {
                int i = subsetIdx[j];
                wrong[j] = (clauses.back().clause.output(trainRawAll[i].x) != trainRawAll[i].y);
                if (wrong[j]) err += w[i];
                wSum += w[i];
            }
            err = max(1e-6, min(0.999, err / wSum));
            double alpha = 0.5 * log((1.0 - err) / err);
            alphas.push_back(alpha);
            for (size_t j = 0; j < subsetIdx.size(); j++) {
                int i = subsetIdx[j];
                w[i] *= wrong[j] ? exp(alpha) : exp(-alpha);
            }
            double norm = 0;
            for (int i = 0; i < Nex; i++) norm += w[i];
            for (int i = 0; i < Nex; i++) w[i] /= norm;
        }

        int correct = 0;
        for (auto& ex : testRawAll) {
            double score = 0;
            for (size_t m = 0; m < clauses.size(); m++) {
                if (!clauses[m].applies(ex.x) || clauses[m].isEmpty()) continue;
                int o = clauses[m].clause.output(ex.x);
                score += alphas[m] * (2.0 * o - 1.0);
            }
            int pred = score > 0 ? 1 : 0;
            if (pred == ex.y) correct++;
        }
        accs[run] = 100.0 * correct / testRawAll.size();

        long totalLit = 0;
        for (auto& c : clauses) for (int i = 0; i < c.clause.n; i++) { if (c.clause.v[i].included()) totalLit++; if (c.clause.vbar[i].included()) totalLit++; }
        cout << "  run " << run << " : acc=" << accs[run] << "%  complexite=" << totalLit << "\n" << flush;
    }
    double mean = 0; for (double a2 : accs) mean += a2; mean /= nbRuns;
    double var = 0; for (double a2 : accs) var += (a2-mean)*(a2-mean); var /= nbRuns;
    cout << which << " : Mean=" << mean << " +/- " << sqrt(var) << "\n";
    return 0;
}
