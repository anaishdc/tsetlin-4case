#include "../tsetlin_engine.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>

// Version multi-classe (one-vs-rest) de chi2_progressive_stump.cpp :
// meme mecanisme (construction progressive de la condition, chi2 local,
// K auto), applique a Iris et Digits.

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

struct ChiStumpClause {
    Clause clause;
    vector<int> condFeatures, condValues;
    vector<bool> active;
    ChiStumpClause(int n, int N, const vector<int>& feats, const vector<int>& vals)
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

struct CandRes { double chi2; int value; };

CandRes evalFeature(const vector<LabeledExample>& data, const vector<int>& subset,
                     const vector<double>& w, int f, double scale) {
    double w00 = 0, w01 = 0, w10 = 0, w11 = 0;
    for (int i : subset) {
        double wi = w[i] * scale;
        int fv = data[i].x[f], y = data[i].y;
        if (fv == 0) { if (y == 0) w00 += wi; else w01 += wi; }
        else         { if (y == 0) w10 += wi; else w11 += wi; }
    }
    double N = w00 + w01 + w10 + w11;
    if (N <= 1e-12) return {0.0, 0};
    double row0 = w00 + w01, row1 = w10 + w11, col0 = w00 + w10, col1 = w01 + w11;
    double obs[2][2] = {{w00, w01}, {w10, w11}};
    double rows[2] = {row0, row1}, cols[2] = {col0, col1};
    double chi2 = 0;
    for (int r = 0; r < 2; r++) for (int c = 0; c < 2; c++) {
        double exp = rows[r] * cols[c] / N;
        if (exp > 1e-9) chi2 += (obs[r][c] - exp) * (obs[r][c] - exp) / exp;
    }
    double pur0 = row0 > 1e-12 ? max(w00, w01) / row0 : 0;
    double pur1 = row1 > 1e-12 ? max(w10, w11) / row1 : 0;
    int value = pur1 >= pur0 ? 1 : 0;
    return {chi2, value};
}

// Meme logique inversee que dans chi2_progressive_stump.cpp : le chi2 dit
// QUAND s'arreter (signal resolvant -> laisser au litteral-learning), pas
// QUELLE feature ajouter (une parite/XOR cachee a un chi2 marginal nul a
// chaque niveau). Sans signal -> avance a l'aveugle (feature aleatoire figee).
void growCondition(const vector<LabeledExample>& data, const vector<double>& w, int n,
                    int maxK, int minSubsetSize, double chi2Threshold, int maxTries,
                    vector<int>& feats, vector<int>& vals) {
    vector<int> subset(data.size());
    iota(subset.begin(), subset.end(), 0);
    vector<bool> used(n, false);
    uniform_int_distribution<int> valDist(0, 1);
    double scale = (double)data.size();
    for (int k = 0; k < maxK; k++) {
        vector<int> pool;
        for (int f = 0; f < n; f++) if (!used[f]) pool.push_back(f);
        if (pool.empty()) break;
        shuffle(pool.begin(), pool.end(), rng);
        int nTry = min((int)pool.size(), maxTries);

        int bestFeat = -1, bestVal = 0; double bestChi2 = -1;
        for (int tries = 0; tries < nTry; tries++) {
            int f = pool[tries];
            auto res = evalFeature(data, subset, w, f, scale);
            if (res.chi2 > bestChi2) { bestChi2 = res.chi2; bestFeat = f; bestVal = res.value; }
        }
        if (bestFeat != -1 && bestChi2 >= chi2Threshold) break; // signal resolvant -> on s'arrete, pas de gel

        int f = pool[0];
        int v = valDist(rng);
        vector<int> newSubset;
        for (int i : subset) if (data[i].x[f] == v) newSubset.push_back(i);
        if ((int)newSubset.size() < minSubsetSize) break;
        feats.push_back(f); vals.push_back(v);
        used[f] = true;
        subset = newSubset;
    }
}

void trainOneClass(const vector<LabeledExample>& trainBin, int n, int N, int M, int total,
                    double S, double a, int maxK, int minSubsetSize, double chi2Threshold, int maxTries,
                    vector<ChiStumpClause>& clauses, vector<double>& alphas) {
    Params p = {S, S, S, S, a, a};
    int Nex = (int)trainBin.size();
    vector<double> w(Nex, 1.0 / Nex);

    for (int m = 0; m < M; m++) {
        vector<int> feats, vals;
        growCondition(trainBin, w, n, maxK, minSubsetSize, chi2Threshold, maxTries, feats, vals);
        clauses.emplace_back(n, N, feats, vals);

        vector<int> subsetIdx;
        for (int i = 0; i < Nex; i++) if (clauses.back().applies(trainBin[i].x)) subsetIdx.push_back(i);
        if (subsetIdx.empty()) { alphas.push_back(0.0); continue; }

        vector<int> subPos, subNeg;
        for (int i : subsetIdx) (trainBin[i].y == 1 ? subPos : subNeg).push_back(i);
        vector<double> wPos, wNeg;
        for (int i : subPos) wPos.push_back(w[i]);
        for (int i : subNeg) wNeg.push_back(w[i]);
        discrete_distribution<int> posDist, negDist;
        if (!wPos.empty()) posDist = discrete_distribution<int>(wPos.begin(), wPos.end());
        if (!wNeg.empty()) negDist = discrete_distribution<int>(wNeg.begin(), wNeg.end());

        for (int t = 0; t < total; t++) {
            bool useNeg = subNeg.empty() ? false : (subPos.empty() ? true : !flip(0.5));
            int idx = useNeg ? subNeg[negDist(rng)] : subPos[posDist(rng)];
            updateMasked(clauses.back().clause, trainBin[idx].x, trainBin[idx].y, p, clauses.back().active);
        }

        double err = 0, wSum = 0;
        vector<bool> wrong(subsetIdx.size());
        for (size_t j = 0; j < subsetIdx.size(); j++) {
            int i = subsetIdx[j];
            wrong[j] = (clauses.back().clause.output(trainBin[i].x) != trainBin[i].y);
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

int main(int argc, char** argv) {
    string which = argc > 1 ? argv[1] : "iris";
    int n, numClasses;
    int N = 100, total = 20000, M = 100, nbRuns = 10;
    double S = 1, a = 0.3;
    int maxK = 4, maxTries = 6;
    double chi2Threshold = 3.84;

    if (which == "iris") {
        n = 16; numClasses = 3;
        M = argc > 2 ? atoi(argv[2]) : 100;
        total = argc > 3 ? atoi(argv[3]) : 20000;
        nbRuns = argc > 4 ? atoi(argv[4]) : 10;
        if (argc > 5) S = atof(argv[5]);
        if (argc > 6) a = atof(argv[6]);
        if (argc > 7) N = atoi(argv[7]);
        if (argc > 8) maxK = atoi(argv[8]);
        int minSubsetSize = argc > 9 ? atoi(argv[9]) : 10;

        cout << which << " (stumps chi2 progressifs, K auto, sans routeur) : M=" << M
             << " total=" << total << " S=" << S << " a=" << a << " N=" << N
             << " maxK=" << maxK << " minSubset=" << minSubsetSize << "\n" << flush;

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

            vector<vector<ChiStumpClause>> ensembles(numClasses);
            vector<vector<double>> alphas(numClasses);
            for (int c = 0; c < numClasses; c++) {
                vector<LabeledExample> trainBin;
                for (auto& e : trainR) trainBin.push_back({e.x, e.y == c ? 1 : 0});
                trainOneClass(trainBin, n, N, M, total, S, a, maxK, minSubsetSize, chi2Threshold, maxTries, ensembles[c], alphas[c]);
            }
            int correct = 0;
            for (auto& ex : testR) {
                vector<double> score(numClasses, 0);
                for (int c = 0; c < numClasses; c++)
                    for (size_t m = 0; m < ensembles[c].size(); m++) {
                        auto& rc = ensembles[c][m];
                        if (!rc.applies(ex.x) || rc.isEmpty()) continue;
                        score[c] += alphas[c][m] * (2.0 * rc.clause.output(ex.x) - 1.0);
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

    // digits
    n = 192; numClasses = 10;
    M = argc > 2 ? atoi(argv[2]) : 100;
    total = argc > 3 ? atoi(argv[3]) : 20000;
    nbRuns = argc > 4 ? atoi(argv[4]) : 5;
    if (argc > 5) S = atof(argv[5]);
    if (argc > 6) a = atof(argv[6]);
    if (argc > 7) N = atoi(argv[7]);
    if (argc > 8) maxK = atoi(argv[8]);
    int minSubsetSize = argc > 9 ? atoi(argv[9]) : 20;

    cout << which << " (stumps chi2 progressifs, K auto, sans routeur, " << nbRuns << " runs) : M=" << M
         << " total=" << total << " S=" << S << " a=" << a << " N=" << N
         << " maxK=" << maxK << " minSubset=" << minSubsetSize << "\n" << flush;

    auto trainRawAll = loadAll("../data/BinaryDigitsTrainingData192.txt", n);
    auto testRawAll  = loadAll("../data/BinaryDigitsTestData192.txt", n);

    vector<double> accs(nbRuns);
    for (int run = 0; run < nbRuns; run++) {
        rng.seed(20000 + run);
        vector<vector<ChiStumpClause>> ensembles(numClasses);
        vector<vector<double>> alphas(numClasses);
        for (int c = 0; c < numClasses; c++) {
            vector<LabeledExample> trainBin;
            for (auto& e : trainRawAll) trainBin.push_back({e.x, e.y == c ? 1 : 0});
            trainOneClass(trainBin, n, N, M, total, S, a, maxK, minSubsetSize, chi2Threshold, maxTries, ensembles[c], alphas[c]);
        }
        int correct = 0;
        for (auto& ex : testRawAll) {
            vector<double> score(numClasses, 0);
            for (int c = 0; c < numClasses; c++)
                for (size_t m = 0; m < ensembles[c].size(); m++) {
                    auto& rc = ensembles[c][m];
                    if (!rc.applies(ex.x) || rc.isEmpty()) continue;
                    score[c] += alphas[c][m] * (2.0 * rc.clause.output(ex.x) - 1.0);
                }
            int pred = 0;
            for (int cc = 1; cc < numClasses; cc++) if (score[cc] > score[pred]) pred = cc;
            if (pred == ex.y) correct++;
        }
        accs[run] = 100.0 * correct / testRawAll.size();
        cout << "  run " << run << " : acc=" << accs[run] << "%\n" << flush;
    }
    double mean = 0; for (double a2 : accs) mean += a2; mean /= nbRuns;
    double var = 0; for (double a2 : accs) var += (a2-mean)*(a2-mean); var /= nbRuns;
    cout << which << " : Mean=" << mean << " +/- " << sqrt(var) << "\n";
    return 0;
}
