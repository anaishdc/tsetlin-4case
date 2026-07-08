#include "../tsetlin_engine.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <functional>
#include <chrono>

// Version multi-classe (one-vs-rest) de fourier_biased_stump.cpp :
// sous-ensemble de features choisi par Fourier pondere (pas au hasard),
// valeur de condition aleatoire, notre regle a 4 cas, chaque clause =
// son propre round de boosting. Pas de routeur.

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

vector<int> findRelevantSubset(const vector<LabeledExample>& data, const vector<double>& w,
                               int n, int maxDegree) {
    int N = (int)data.size();
    double wTotal = 0;
    vector<double> ys(N);
    for (int i = 0; i < N; i++) { ys[i] = 2.0 * data[i].y - 1.0; wTotal += w[i]; }

    auto evalSubset = [&](const vector<int>& subset) {
        double sum = 0;
        for (int i = 0; i < N; i++) {
            double prod = ys[i] * w[i];
            for (int j : subset) prod *= (2 * data[i].x[j] - 1);
            sum += prod;
        }
        return sum / wTotal;
    };

    double bestAbs = -1; vector<int> bestSubset;
    for (int deg = 1; deg <= maxDegree; deg++) {
        vector<int> idx(deg);
        function<void(int,int)> rec = [&](int start, int depth) {
            if (depth == deg) {
                double c = evalSubset(idx);
                if (fabs(c) > bestAbs) { bestAbs = fabs(c); bestSubset = idx; }
                return;
            }
            for (int f = start; f < n; f++) { idx[depth] = f; rec(f + 1, depth + 1); }
        };
        rec(0, 0);
    }
    return bestSubset;
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

void trainOneClass(const vector<LabeledExample>& trainBin, int n, int N, int M, int maxDegree, int total,
                   double S, double a, vector<RandomStumpClause>& clauses, vector<double>& alphas) {
    Params p = {S, S, S, S, a, a};
    uniform_int_distribution<int> valDist(0, 1);
    int Nex = (int)trainBin.size();
    vector<double> w(Nex, 1.0 / Nex);

    for (int m = 0; m < M; m++) {
        auto S_subset = findRelevantSubset(trainBin, w, n, maxDegree);
        vector<int> feats, vals;
        for (size_t k = 0; k + 1 < S_subset.size(); k++) feats.push_back(S_subset[k]);
        // parcourt systematiquement les combinaisons de valeurs (pas de tirage
        // aleatoire) : le round m essaie la combinaison numero (m % 2^|feats|)
        int nCombos = 1 << feats.size();
        int combo = m % nCombos;
        for (size_t k = 0; k < feats.size(); k++) vals.push_back((combo >> k) & 1);
        clauses.emplace_back(n, N, feats, vals);

        vector<int> subsetIdx;
        for (int i = 0; i < Nex; i++) if (clauses.back().applies(trainBin[i].x)) subsetIdx.push_back(i);
        int minSubset = max(10, Nex / 20);  // evite les sous-ensembles trop petits (surapprentissage + alpha gonfle)
        if ((int)subsetIdx.size() < minSubset) { alphas.push_back(0.0); continue; }

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
        err = max(0.02, min(0.98, err / wSum));  // plancher a 2% : evite l'alpha demesure meme pour un sous-ensemble licite mais petit
        double alpha = 0.5 * log((1.0 - err) / err);
        alphas.push_back(alpha);
        cerr << "    [diag] round " << m << " : subset={";
        for (size_t k=0;k<S_subset.size();k++) cerr << (k?",":"") << S_subset[k];
        cerr << "} subsetSize=" << subsetIdx.size() << "/" << Nex << " err=" << err << " alpha=" << alpha << "\n";
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
    int N = 100, total = 20000, M = 20, nbRuns = 10, maxDegree = 2;
    double S = 1, a = 0.3;

    if (which == "iris") {
        n = 16; numClasses = 3;
        M = argc > 2 ? atoi(argv[2]) : 20;
        maxDegree = argc > 3 ? atoi(argv[3]) : 2;
        total = argc > 4 ? atoi(argv[4]) : 20000;
        nbRuns = argc > 5 ? atoi(argv[5]) : 10;
        if (argc > 6) S = atof(argv[6]);
        if (argc > 7) a = atof(argv[7]);
        if (argc > 8) N = atoi(argv[8]);

        cout << which << " (Fourier-biais + notre regle, sans routeur) : M=" << M << " maxDegree=" << maxDegree
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
                trainOneClass(trainBin, n, N, M, maxDegree, total, S, a, ensembles[c], alphas[c]);
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

    // digits (split fixe)
    n = 192; numClasses = 10;
    M = argc > 2 ? atoi(argv[2]) : 20;
    maxDegree = argc > 3 ? atoi(argv[3]) : 1;
    total = argc > 4 ? atoi(argv[4]) : 20000;
    if (argc > 6) S = atof(argv[6]);
    if (argc > 7) a = atof(argv[7]);
    if (argc > 8) N = atoi(argv[8]);

    cout << which << " (Fourier-biais + notre regle, sans routeur) : M=" << M << " maxDegree=" << maxDegree
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
        trainOneClass(trainBin, n, N, M, maxDegree, total, S, a, ensembles[c], alphas[c]);
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
                score[c] += alphas[c][m] * (2.0 * rc.clause.output(ex.x) - 1.0);
            }
        int pred = 0;
        for (int cc = 1; cc < numClasses; cc++) if (score[cc] > score[pred]) pred = cc;
        if (pred == ex.y) correct++;
    }
    double acc = 100.0 * correct / testRawAll.size();
    cout << which << " -- accuracy=" << acc << "%\n";
    return 0;
}
