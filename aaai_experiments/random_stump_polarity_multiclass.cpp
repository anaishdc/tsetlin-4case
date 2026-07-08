#include "../tsetlin_engine.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <chrono>

// Version multi-classe (one-vs-rest) de random_stump_polarity.cpp.

vector<LabeledExample> loadAll(const string& path, int n) {
    vector<LabeledExample> data;
    ifstream f(path); string line;
    while (getline(f, line)) {
        if (line.empty()) continue;
        istringstream iss(line);
        vector<int> x(n);
        for (int i = 0; i < n; i++) iss >> x[i];
        int label; iss >> label;
        data.push_back({x, label});
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

void trainOneClass(const vector<LabeledExample>& trainBin, int n, int N, int M, int total,
                   double S, double a, vector<RandomStumpClause>& clauses, vector<double>& alphas) {
    Params p = {S, S, S, S, a, a};
    uniform_int_distribution<int> featDist(0, n - 1);
    uniform_int_distribution<int> valDist(0, 1);
    uniform_int_distribution<int> polDist(0, 1);
    int Nex = (int)trainBin.size();
    vector<double> w(Nex, 1.0 / Nex);

    for (int m = 0; m < M; m++) {
        int f = featDist(rng), v = valDist(rng), pol = polDist(rng) == 0 ? +1 : -1;
        clauses.emplace_back(n, N, f, v, pol);

        vector<int> subsetIdx;
        for (int i = 0; i < Nex; i++) if (clauses.back().applies(trainBin[i].x)) subsetIdx.push_back(i);
        if (subsetIdx.empty()) { alphas.push_back(0.0); continue; }

        vector<int> subPos, subNeg;
        for (int i : subsetIdx) (clauses.back().localTarget(trainBin[i].y) == 1 ? subPos : subNeg).push_back(i);
        vector<double> wPos, wNeg;
        for (int i : subPos) wPos.push_back(w[i]);
        for (int i : subNeg) wNeg.push_back(w[i]);
        discrete_distribution<int> posDist, negDist;
        if (!wPos.empty()) posDist = discrete_distribution<int>(wPos.begin(), wPos.end());
        if (!wNeg.empty()) negDist = discrete_distribution<int>(wNeg.begin(), wNeg.end());

        for (int t = 0; t < total; t++) {
            bool useNeg = subNeg.empty() ? false : (subPos.empty() ? true : !flip(0.5));
            int idx = useNeg ? subNeg[negDist(rng)] : subPos[posDist(rng)];
            int localY = clauses.back().localTarget(trainBin[idx].y);
            updateMasked(clauses.back().clause, trainBin[idx].x, localY, p, clauses.back().active);
        }

        double err = 0, wSum = 0;
        vector<bool> wrong(subsetIdx.size());
        for (size_t j = 0; j < subsetIdx.size(); j++) {
            int i = subsetIdx[j];
            int localY = clauses.back().localTarget(trainBin[i].y);
            wrong[j] = (clauses.back().clause.output(trainBin[i].x) != localY);
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
}

double clauseScore(const RandomStumpClause& rc, const vector<int>& x) {
    int o = rc.clause.output(x);
    int vote = (rc.polarity > 0) ? o : (1 - o);
    return 2.0 * vote - 1.0;
}

int main(int argc, char** argv) {
    string which = argc > 1 ? argv[1] : "iris";
    int n, numClasses;
    int N = 100, total = 20000, M = 100, nbRuns = 10;
    double S = 1, a = 0.3;

    if (which == "iris") {
        n = 16; numClasses = 3;
        M = argc > 2 ? atoi(argv[2]) : 100;
        total = argc > 3 ? atoi(argv[3]) : 20000;
        nbRuns = argc > 4 ? atoi(argv[4]) : 10;
        if (argc > 5) S = atof(argv[5]);
        if (argc > 6) a = atof(argv[6]);
        if (argc > 7) N = atoi(argv[7]);

        cout << which << " (stumps aleatoires + polarite, notre regle) : M=" << M
             << " total=" << total << " S=" << S << " a=" << a << " N=" << N << "\n" << flush;

        auto allRawFull = loadAll("../data/BinaryIrisData.txt", n);
        vector<double> accs(nbRuns);
        for (int run = 0; run < nbRuns; run++) {
            mt19937 shufRng(20000 + run);
            auto allRaw = allRawFull;
            shuffle(allRaw.begin(), allRaw.end(), shufRng);
            int trainSize = (int)round(0.8 * allRaw.size());
            vector<LabeledExample> trainR(allRaw.begin(), allRaw.begin() + trainSize);
            vector<LabeledExample> testR(allRaw.begin() + trainSize, allRaw.end());
            rng.seed(20000 + run);

            vector<vector<RandomStumpClause>> ensembles(numClasses);
            vector<vector<double>> alphas(numClasses);
            for (int c = 0; c < numClasses; c++) {
                vector<LabeledExample> trainBin;
                for (auto& e : trainR) trainBin.push_back({e.x, e.y == c ? 1 : 0});
                trainOneClass(trainBin, n, N, M, total, S, a, ensembles[c], alphas[c]);
            }
            int correct = 0;
            for (auto& ex : testR) {
                vector<double> score(numClasses, 0);
                for (int c = 0; c < numClasses; c++)
                    for (size_t m = 0; m < ensembles[c].size(); m++) {
                        auto& rc = ensembles[c][m];
                        if (!rc.applies(ex.x) || rc.isEmpty()) continue;
                        score[c] += alphas[c][m] * clauseScore(rc, ex.x);
                    }
                int pred = 0;
                for (int cc = 1; cc < numClasses; cc++) if (score[cc] > score[pred]) pred = cc;
                if (pred == ex.y) correct++;
            }
            accs[run] = 100.0 * correct / testR.size();
            cout << "  run " << run << " : acc=" << accs[run] << "%\n" << flush;
        }
        double mean = 0; for (double a2 : accs) mean += a2; mean /= nbRuns;
        double var = 0; for (double a2 : accs) var += (a2-mean)*(a2-mean); var /= nbRuns;
        cout << which << " : Mean=" << mean << " +/- " << sqrt(var) << "\n";
        return 0;
    }

    // digits (split fixe)
    n = 192; numClasses = 10;
    M = argc > 2 ? atoi(argv[2]) : 100;
    total = argc > 3 ? atoi(argv[3]) : 20000;
    if (argc > 4) S = atof(argv[4]);
    if (argc > 5) a = atof(argv[5]);
    if (argc > 6) N = atoi(argv[6]);

    cout << which << " (stumps aleatoires + polarite, notre regle) : M=" << M
         << " total=" << total << " S=" << S << " a=" << a << " N=" << N << "\n" << flush;

    auto trainRawAll = loadAll("../data/BinaryDigitsTrainingData192.txt", n);
    auto testRawAll  = loadAll("../data/BinaryDigitsTestData192.txt", n);
    rng.seed(20000);

    vector<vector<RandomStumpClause>> ensembles(numClasses);
    vector<vector<double>> alphas(numClasses);
    for (int c = 0; c < numClasses; c++) {
        auto t0 = chrono::steady_clock::now();
        vector<LabeledExample> trainBin;
        for (auto& e : trainRawAll) trainBin.push_back({e.x, e.y == c ? 1 : 0});
        trainOneClass(trainBin, n, N, M, total, S, a, ensembles[c], alphas[c]);
        auto t1 = chrono::steady_clock::now();
        cout << "  classe " << c << " terminee [" << chrono::duration<double>(t1-t0).count() << "s]\n" << flush;
    }
    int correct = 0;
    for (auto& ex : testRawAll) {
        vector<double> score(numClasses, 0);
        for (int c = 0; c < numClasses; c++)
            for (size_t m = 0; m < ensembles[c].size(); m++) {
                auto& rc = ensembles[c][m];
                if (!rc.applies(ex.x) || rc.isEmpty()) continue;
                score[c] += alphas[c][m] * clauseScore(rc, ex.x);
            }
        int pred = 0;
        for (int cc = 1; cc < numClasses; cc++) if (score[cc] > score[pred]) pred = cc;
        if (pred == ex.y) correct++;
    }
    double acc = 100.0 * correct / testRawAll.size();
    cout << which << " -- accuracy=" << acc << "%\n";
    return 0;
}
