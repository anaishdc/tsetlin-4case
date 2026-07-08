#include "../tsetlin_engine.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>

// Comme random_stump_perclause_boost.cpp, mais chaque clause recoit
// EN PLUS une polarite aleatoire (+1 ou -1) : polarite + entraine
// vers y directement, polarite - entraine vers 1-y (detecte le
// motif "classe negative"). Chaque clause reste son propre round
// de boosting independant. Le score final tient compte de la
// polarite dans la combinaison.

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
    int condFeature, condValue, polarity;
    vector<bool> active;
    RandomStumpClause(int n, int N, int f, int v, int pol) : clause(n, N), condFeature(f), condValue(v), polarity(pol), active(n, true) {
        clause.v[f].state = 1; clause.vbar[f].state = 1;
        active[f] = false;
    }
    bool applies(const vector<int>& x) const { return x[condFeature] == condValue; }
    bool isEmpty() const {
        for (int i = 0; i < clause.n; i++) if (clause.v[i].included() || clause.vbar[i].included()) return false;
        return true;
    }
    int localTarget(int y) const { return polarity > 0 ? y : 1 - y; }
};

int main(int argc, char** argv) {
    string which = argc > 1 ? argv[1] : "xor";
    int n; vector<LabeledExample> trainRawAll, testRawAll;
    int N = 100, total = 1500, M = 100, nbRuns = 10;
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
    nbRuns = argc > 3 ? atoi(argv[3]) : 10;
    total = argc > 4 ? atoi(argv[4]) : 1500;
    if (argc > 5) S = atof(argv[5]);
    if (argc > 6) a = atof(argv[6]);
    if (argc > 7) N = atoi(argv[7]);

    cout << which << " (stumps aleatoires + polarite +/-, notre regle) : M=" << M
         << " total/clause=" << total << " S=" << S << " a=" << a << " N=" << N << "\n" << flush;

    Params p = {S, S, S, S, a, a};
    uniform_int_distribution<int> featDist(0, n - 1);
    uniform_int_distribution<int> valDist(0, 1);
    uniform_int_distribution<int> polDist(0, 1);

    vector<double> accs(nbRuns);
    for (int run = 0; run < nbRuns; run++) {
        rng.seed(180000 + run);
        int Nex = (int)trainRawAll.size();
        vector<double> w(Nex, 1.0 / Nex);
        vector<RandomStumpClause> clauses;
        vector<double> alphas;

        for (int m = 0; m < M; m++) {
            int f = featDist(rng), v = valDist(rng), pol = polDist(rng) == 0 ? +1 : -1;
            clauses.emplace_back(n, N, f, v, pol);

            vector<int> subsetIdx;
            for (int i = 0; i < Nex; i++) if (clauses.back().applies(trainRawAll[i].x)) subsetIdx.push_back(i);
            if (subsetIdx.empty()) { alphas.push_back(0.0); continue; }

            // pos/neg definis par rapport a la cible LOCALE (apres polarite)
            vector<int> subPos, subNeg;
            for (int i : subsetIdx) (clauses.back().localTarget(trainRawAll[i].y) == 1 ? subPos : subNeg).push_back(i);
            vector<double> wPos, wNeg;
            for (int i : subPos) wPos.push_back(w[i]);
            for (int i : subNeg) wNeg.push_back(w[i]);
            discrete_distribution<int> posDist, negDist;
            if (!wPos.empty()) posDist = discrete_distribution<int>(wPos.begin(), wPos.end());
            if (!wNeg.empty()) negDist = discrete_distribution<int>(wNeg.begin(), wNeg.end());

            for (int t = 0; t < total; t++) {
                bool useNeg = subNeg.empty() ? false : (subPos.empty() ? true : !flip(0.5));
                int idx = useNeg ? subNeg[negDist(rng)] : subPos[posDist(rng)];
                int localY = clauses.back().localTarget(trainRawAll[idx].y);
                updateMasked(clauses.back().clause, trainRawAll[idx].x, localY, p, clauses.back().active);
            }

            double err = 0, wSum = 0;
            vector<bool> wrong(subsetIdx.size());
            for (size_t j = 0; j < subsetIdx.size(); j++) {
                int i = subsetIdx[j];
                int localY = clauses.back().localTarget(trainRawAll[i].y);
                wrong[j] = (clauses.back().clause.output(trainRawAll[i].x) != localY);
                if (wrong[j]) err += w[i];
                wSum += w[i];
            }
            err = max(1e-6, min(0.999, err / wSum));
            double alpha = 0.5 * log((1.0 - err) / err);
            alphas.push_back(alpha);
            double norm = 0;
            for (size_t j = 0; j < subsetIdx.size(); j++) {
                int i = subsetIdx[j];
                w[i] *= wrong[j] ? exp(alpha) : exp(-alpha);
            }
            for (int i = 0; i < Nex; i++) norm += w[i];
            for (int i = 0; i < Nex; i++) w[i] /= norm;
        }

        int correct = 0;
        for (auto& ex : testRawAll) {
            double score = 0;
            for (size_t m = 0; m < clauses.size(); m++) {
                if (!clauses[m].applies(ex.x) || clauses[m].isEmpty()) continue;
                int o = clauses[m].clause.output(ex.x);   // o=1 signifie "localTarget detecte"
                int vote = (clauses[m].polarity > 0) ? o : (1 - o); // ramene au sens de y=1
                score += alphas[m] * (2.0 * vote - 1.0);
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
