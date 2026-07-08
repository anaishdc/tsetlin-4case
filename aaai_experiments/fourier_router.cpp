#include "../tsetlin_engine.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <chrono>

// -----------------------------------------------------------------
// Remplace UNIQUEMENT le routeur (arbre + Gini + lookahead) par une
// detection directe des variables pertinentes via corrélation de
// Fourier -- la clause (automates de Tsetlin, regle a 4 cas) reste
// EXACTEMENT la meme que dans tsetlin_engine.h. Chaque round de
// boosting : Fourier trouve le sous-ensemble S le plus correle
// (pondere), puis une Clause standard est entrainee dessus (les
// features hors S sont gelees/exclues, comme le fait le routeur
// normalement pour les features utilisees dans le chemin).
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

// Trouve le sous-ensemble de variables le plus correle (Fourier
// pondere), degre 1..maxDegree, avec extension gloutonne optionnelle.
vector<int> findRelevantSubset(const vector<LabeledExample>& data, const vector<double>& w,
                               int n, int maxDegree, int targetSize) {
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
        if (bestAbs > 0.5) break;
    }

    // extension gloutonne (cout O(n) par variable ajoutee)
    if (targetSize > (int)bestSubset.size()) {
        vector<bool> used(n, false);
        for (int f : bestSubset) used[f] = true;
        vector<double> curProd(N);
        for (int i = 0; i < N; i++) {
            double p = ys[i] * w[i];
            for (int f : bestSubset) p *= (2 * data[i].x[f] - 1);
            curProd[i] = p;
        }
        while ((int)bestSubset.size() < targetSize) {
            int bestF = -1; double bestGain = -1;
            for (int f = 0; f < n; f++) {
                if (used[f]) continue;
                double sum = 0;
                for (int i = 0; i < N; i++) sum += curProd[i] * (2 * data[i].x[f] - 1);
                double c = fabs(sum / wTotal);
                if (c > bestGain) { bestGain = c; bestF = f; }
            }
            if (bestF == -1) break;
            used[bestF] = true;
            bestSubset.push_back(bestF);
            for (int i = 0; i < N; i++) curProd[i] *= (2 * data[i].x[bestF] - 1);
        }
    }
    return bestSubset;
}

struct WeightedSampler {
    const vector<LabeledExample>& data;
    vector<int> posIdx, negIdx;
    discrete_distribution<int> posDist, negDist;
    WeightedSampler(const vector<LabeledExample>& d, const vector<double>& w) : data(d) {
        for (int i = 0; i < (int)d.size(); i++) (d[i].y == 1 ? posIdx : negIdx).push_back(i);
        vector<double> wPos, wNeg;
        for (int i : posIdx) wPos.push_back(w[i]);
        for (int i : negIdx) wNeg.push_back(w[i]);
        if (!wPos.empty()) posDist = discrete_distribution<int>(wPos.begin(), wPos.end());
        if (!wNeg.empty()) negDist = discrete_distribution<int>(wNeg.begin(), wNeg.end());
    }
    const LabeledExample& sample() {
        bool useNeg = negIdx.empty() ? false : (posIdx.empty() ? true : !flip(0.5));
        if (useNeg) return data[negIdx[negDist(rng)]];
        return data[posIdx[posDist(rng)]];
    }
};

// Une "feuille Fourier" : la MEME Clause standard (automates + regle a
// 4 cas), mais dont les features hors du sous-ensemble trouve par
// Fourier sont gelees (state=1, exclu, comme le fait le routeur).
// Routeur mini : coupe SEQUENTIELLEMENT sur chaque variable du
// sous-ensemble trouve par Fourier (au lieu du choix de split par
// Gini/lookahead) -- reconstruit la structure multi-feuilles
// necessaire pour representer XOR/parite (chaque feuille garde un
// residu simple, representable par UNE seule clause conjonctive).
struct FourierRouterLeaf {
    Clause clause;
    Params params;
    vector<bool> active;
    FourierRouterLeaf(int n, int N, const vector<bool>& usedFeatures, double baseS, double baseA)
        : clause(n, N), active(n, true) {
        for (int i = 0; i < n; i++) if (usedFeatures[i]) { active[i] = false; clause.v[i].state = 1; clause.vbar[i].state = 1; }
        params = {baseS, baseS, baseS, baseS, baseA, baseA};
    }
};

struct FourierRouterNet {
    vector<int> splitVars;              // ordre des variables de coupure
    vector<unique_ptr<FourierRouterLeaf>> leaves; // 2^|splitVars| feuilles
    vector<double> leafPurity;          // baseline : vote majoritaire pondere de la feuille (sans clause)

    int route(const vector<int>& x) const {
        int idx = 0;
        for (int f : splitVars) idx = idx * 2 + x[f];
        return idx;
    }
    int output(const vector<int>& x) const { return leaves[route(x)]->clause.output(x); }
    double baselineScore(const vector<int>& x) const { return leafPurity[route(x)] - 0.5; }
    double normalizedScore(const vector<int>& x) const {
        const Clause& c = leaves[route(x)]->clause;
        int satisfied = 0, violated = 0, nLit = 0;
        for (int i = 0; i < c.n; i++) {
            if (c.v[i].included())    { nLit++; x[i] == 1 ? satisfied++ : violated++; }
            if (c.vbar[i].included()) { nLit++; x[i] == 0 ? satisfied++ : violated++; }
        }
        int rawScore = violated > 0 ? -violated : satisfied;
        return nLit > 0 ? (double)rawScore / nLit : 0.0;
    }
    void update(const vector<int>& x, int y) {
        auto& leaf = *leaves[route(x)];
        updateMasked(leaf.clause, x, y, leaf.params, leaf.active);
    }
};

FourierRouterNet buildFourierRouter(const vector<LabeledExample>& data, const vector<double>& w,
                                    int n, int N, int maxDegree, int splitDepth, double baseS, double baseA) {
    auto S = findRelevantSubset(data, w, n, maxDegree, splitDepth); // extension gloutonne jusqu'a splitDepth variables
    if (splitDepth >= 0 && (int)S.size() > splitDepth) S.resize(splitDepth); // securite si l'exhaustif seul depasse deja splitDepth
    FourierRouterNet net;
    net.splitVars = S;
    vector<bool> usedFeatures(n, false);
    for (int f : S) usedFeatures[f] = true;
    int K = 1 << S.size();
    for (int k = 0; k < K; k++) net.leaves.push_back(make_unique<FourierRouterLeaf>(n, N, usedFeatures, baseS, baseA));

    net.leafPurity.assign(K, 0.5);
    vector<double> wSum(K, 0), wPos(K, 0);
    for (size_t i = 0; i < data.size(); i++) {
        int idx = 0;
        for (int f : S) idx = idx * 2 + data[i].x[f];
        wSum[idx] += w[i];
        if (data[i].y == 1) wPos[idx] += w[i];
    }
    for (int k = 0; k < K; k++) if (wSum[k] > 0) net.leafPurity[k] = wPos[k] / wSum[k];
    return net;
}

void trainBoosted(const vector<LabeledExample>& trainBin, int n, int maxDegree, int splitDepth,
                  int N, double baseS, double baseA, int total, int M,
                  vector<FourierRouterNet>& rounds, vector<double>& alphas) {
    int Nex = (int)trainBin.size();
    vector<double> w(Nex, 1.0 / Nex);
    for (int m = 0; m < M; m++) {
        rounds.push_back(buildFourierRouter(trainBin, w, n, N, maxDegree, splitDepth, baseS, baseA));

        WeightedSampler sampler(trainBin, w);
        for (int t = 0; t < total; t++) { const auto& ex = sampler.sample(); rounds.back().update(ex.x, ex.y); }

        double err = 0, wSum = 0;
        vector<bool> wrong(Nex);
        for (int i = 0; i < Nex; i++) {
            wrong[i] = (rounds.back().output(trainBin[i].x) != trainBin[i].y);
            if (wrong[i]) err += w[i];
            wSum += w[i];
        }
        err = max(1e-6, min(0.999, err / wSum));
        double alpha = 0.5 * log((1.0 - err) / err);
        alphas.push_back(alpha);
        double norm = 0;
        for (int i = 0; i < Nex; i++) { w[i] *= wrong[i] ? exp(alpha) : exp(-alpha); norm += w[i]; }
        for (int i = 0; i < Nex; i++) w[i] /= norm;
    }
}

int main(int argc, char** argv) {
    string which = argc > 1 ? argv[1] : "xor";
    int n, numClasses = 1, maxDegree, M, splitDepth;
    vector<LabeledExample> trainRawAll, testRawAll;
    int N = 500, total = 30000; double baseS = 1, baseA = 0.3;

    if (which == "xor") {
        n = 12; maxDegree = 2; splitDepth = 1; M = argc > 2 ? atoi(argv[2]) : 5;
        trainRawAll = loadDataset("../data/NoisyXORTrainingData.txt", n);
        testRawAll  = loadDataset("../data/NoisyXORTestData.txt", n);
    } else if (which == "mux11") {
        n = 11; maxDegree = 3; splitDepth = 3; M = argc > 2 ? atoi(argv[2]) : 5;
        trainRawAll = loadDataset("../data/Mux11TrainingData.txt", n);
        testRawAll  = loadDataset("../data/Mux11TestData.txt", n);
    } else { // parity3
        n = 12; maxDegree = 3; splitDepth = 2; M = argc > 2 ? atoi(argv[2]) : 5;
        trainRawAll = loadDataset("../data/Parity3TrainingData.txt", n);
        testRawAll  = loadDataset("../data/Parity3TestData.txt", n);
    }
    if (argc > 3) maxDegree = atoi(argv[3]);
    if (argc > 5) splitDepth = atoi(argv[5]);
    int nbRuns = argc > 4 ? atoi(argv[4]) : 10;

    cout << which << " (Fourier-router + clause standard) : maxDegree=" << maxDegree
         << " M=" << M << " splitDepth=" << splitDepth << "\n" << flush;

    vector<double> accs(nbRuns), accsBase(nbRuns);
    for (int run = 0; run < nbRuns; run++) {
        rng.seed(90000 + run);
        vector<FourierRouterNet> rounds;
        vector<double> alphas;
        trainBoosted(trainRawAll, n, maxDegree, splitDepth, N, baseS, baseA, total, M, rounds, alphas);

        int correct = 0, correctBase = 0;
        for (auto& ex : testRawAll) {
            double score = 0, scoreBase = 0;
            for (size_t m = 0; m < rounds.size(); m++) {
                score += alphas[m] * rounds[m].normalizedScore(ex.x);
                scoreBase += alphas[m] * rounds[m].baselineScore(ex.x);
            }
            int pred = score > 0 ? 1 : 0;
            int predBase = scoreBase > 0 ? 1 : 0;
            if (pred == ex.y) correct++;
            if (predBase == ex.y) correctBase++;
        }
        accs[run] = 100.0 * correct / testRawAll.size();
        accsBase[run] = 100.0 * correctBase / testRawAll.size();
        cout << "  run " << run << " : systeme_complet=" << accs[run]
             << "%  baseline_routeur_seul=" << accsBase[run] << "%\n" << flush;
    }
    auto stats = [&](vector<double>& v, const string& label) {
        double mean = 0; for (double a : v) mean += a; mean /= nbRuns;
        double var = 0; for (double a : v) var += (a-mean)*(a-mean); var /= nbRuns;
        cout << label << " : Mean=" << mean << " +/- " << sqrt(var) << "\n";
    };
    stats(accs, which + " systeme complet");
    stats(accsBase, which + " baseline (routeur Fourier seul, sans clause)");
    return 0;
}
