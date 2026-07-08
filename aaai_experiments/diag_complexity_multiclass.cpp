#include "../tsetlin_engine.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>

// Diagnostic de complexite pour Iris/Digits (multi-classe, one-vs-rest).
// Usage : ./diag_complexity_multiclass <iris|digits>

vector<LabeledExample> loadAll(const string& path, int n) {
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

vector<double> leafPuritiesWeighted(const RouterTree& tree, const vector<LabeledExample>& trainBin, const vector<double>& w) {
    int K = (int)tree.leafUsedFeatures.size();
    vector<double> wSum(K, 0.0), wPos(K, 0.0);
    for (size_t i = 0; i < trainBin.size(); i++) {
        int leaf = tree.route(trainBin[i].x);
        wSum[leaf] += w[i];
        if (trainBin[i].y == 1) wPos[leaf] += w[i];
    }
    vector<double> purity(K, 0.0);
    for (int k = 0; k < K; k++) purity[k] = wSum[k] > 0 ? wPos[k] / wSum[k] : 0.0;
    return purity;
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

void trainBoostedClass(const vector<LabeledExample>& trainBin, int n, int K, int N,
                       double S, double a, int total, int discoverCount, double noiseTol, int M,
                       vector<MultiClauseNetwork>& nets, vector<double>& alphas, vector<vector<double>>& purities) {
    int Nex = (int)trainBin.size();
    vector<double> w(Nex, 1.0 / Nex);
    for (int m = 0; m < M; m++) {
        vector<int> idx(Nex);
        for (int i = 0; i < Nex; i++) idx[i] = i;
        sort(idx.begin(), idx.end(), [&](int a2, int b2) { return w[a2] > w[b2]; });
        int dc = min(Nex, discoverCount);
        vector<WeightedLabeledExample> discoverBatch;
        for (int i = 0; i < dc; i++) discoverBatch.push_back({ trainBin[idx[i]].x, trainBin[idx[i]].y, w[idx[i]] });
        auto router = buildWeightedRouterTree(discoverBatch, n, K, noiseTol);
        purities.push_back(leafPuritiesWeighted(*router, trainBin, w));
        nets.emplace_back(n, N, K, std::move(router), S, a);
        WeightedSampler sampler(trainBin, w);
        for (int t = 0; t < total; t++) { const auto& ex = sampler.sample(); nets.back().update(ex.x, ex.y); }
        double err = 0, wSum = 0;
        vector<bool> wrong(Nex);
        for (int i = 0; i < Nex; i++) {
            wrong[i] = (nets.back().output(trainBin[i].x) != trainBin[i].y);
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

int routerDepth(const RouterTree& tree, const vector<int>& x) {
    RouterNode* node = tree.root.get();
    int depth = 0;
    while (!node->isLeaf) { node = (x[node->splitFeature] == 0) ? node->left.get() : node->right.get(); depth++; }
    return depth;
}
int includedLiterals(const Clause& c) {
    int n = 0;
    for (int i = 0; i < c.n; i++) { if (c.v[i].included()) n++; if (c.vbar[i].included()) n++; }
    return n;
}

int main(int argc, char** argv) {
    string which = argc > 1 ? argv[1] : "iris";
    int n, K, N, M, numClasses;
    double noiseTol;
    string dataPath, testPathOpt;
    bool reshuffleSplit = false;

    vector<LabeledExample> allRaw, trainRaw, testRaw;

    if (which == "iris") {
        n = 16; K = 2; N = 320; M = 16; numClasses = 3; noiseTol = 0.10;
        allRaw = loadAll("../data/BinaryIrisData.txt", n);
        reshuffleSplit = true;
    } else { // digits
        n = 192; K = 4; N = 1000; M = 16; numClasses = 10; noiseTol = 0.05;
        trainRaw = loadAll("../data/BinaryDigitsTrainingData192.txt", n);
        testRaw  = loadAll("../data/BinaryDigitsTestData192.txt", n);
    }
    int total = 100000, discoverCount = max(300, 3*n);

    if (reshuffleSplit) {
        rng.seed(20000);
        vector<LabeledExample> shuffled = allRaw;
        shuffle(shuffled.begin(), shuffled.end(), rng);
        int trainSize = (int)std::round(0.8 * shuffled.size());
        trainRaw = vector<LabeledExample>(shuffled.begin(), shuffled.begin() + trainSize);
        testRaw  = vector<LabeledExample>(shuffled.begin() + trainSize, shuffled.end());
    } else {
        rng.seed(20000);
    }

    vector<vector<MultiClauseNetwork>> ensembles(numClasses);
    vector<vector<double>> alphas(numClasses);
    vector<vector<vector<double>>> purities(numClasses);
    for (int c = 0; c < numClasses; c++) {
        vector<LabeledExample> trainBin;
        for (auto& e : trainRaw) trainBin.push_back({ e.x, e.y == c ? 1 : 0 });
        trainBoostedClass(trainBin, n, K, N, 1, 0.3, total, discoverCount, noiseTol, M,
                          ensembles[c], alphas[c], purities[c]);
    }

    // 1) complexite totale (tout l'ensemble, toutes classes)
    long totalLit = 0, totalNodes = 0;
    for (int c = 0; c < numClasses; c++)
        for (auto& net : ensembles[c]) {
            for (auto& cl : net.clauses) totalLit += includedLiterals(cl);
            totalNodes += (long)net.router->leafUsedFeatures.size() - 1;
        }
    cout << which << " -- complexite totale (litteraux Include, " << numClasses << " classes x M=" << M << ") : " << totalLit << "\n";
    cout << which << " -- noeuds routeur totaux : " << totalNodes << "\n";

    // 2) complexite de decision moyenne par prediction (toutes classes evaluees pour argmax)
    double avgComplexity = 0;
    for (auto& ex : testRaw) {
        int complexity = 0;
        for (int c = 0; c < numClasses; c++)
            for (auto& net : ensembles[c]) {
                int leaf = net.router->route(ex.x);
                complexity += routerDepth(*net.router, ex.x) + includedLiterals(net.clauses[leaf]);
            }
        avgComplexity += complexity;
    }
    avgComplexity /= testRaw.size();
    cout << which << " -- complexite de decision moyenne / prediction : " << avgComplexity << "\n";

    return 0;
}
