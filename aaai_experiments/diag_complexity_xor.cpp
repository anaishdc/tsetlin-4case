#include "../tsetlin_engine.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>

// Diagnostic de complexite pour XOR : notre systeme deja valide
// (K=2, M=5). Calcule :
//   1) complexite totale (nb de litteraux Include, somme sur tout l'ensemble)
//   2) complexite de decision (nb d'operations pour classer UN exemple :
//      profondeur du routeur + littéraux de la feuille atteinte, somme sur les M rounds)

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

int main() {
    int n = 12, K = 2, N = 500, total = 30000, M = 5;
    double noiseTol = 0.40, discoverCount = max(300, 3*n);
    auto trainData = loadDataset("../data/NoisyXORTrainingData.txt", n);
    auto testData  = loadDataset("../data/NoisyXORTestData.txt", n);

    rng.seed(30000);
    vector<MultiClauseNetwork> nets;
    vector<double> alphas;
    vector<vector<double>> purities;
    trainBoostedClass(trainData, n, K, N, 1, 0.3, total, (int)discoverCount, noiseTol, M, nets, alphas, purities);

    // 1) complexite totale (tout l'ensemble)
    long totalLit = 0, totalNodes = 0;
    for (auto& net : nets) {
        for (auto& c : net.clauses) totalLit += includedLiterals(c);
        // nombre de noeuds internes du routeur = nb feuilles - 1
        totalNodes += (long)net.router->leafUsedFeatures.size() - 1;
    }
    cout << "XOR -- complexite totale (litteraux Include, somme M=" << M << " rounds) : " << totalLit << "\n";
    cout << "XOR -- noeuds routeur totaux (M rounds) : " << totalNodes << "\n";

    // 2) complexite de decision moyenne (par prediction, sur le test set)
    double avgComplexity = 0;
    for (auto& ex : testData) {
        int complexity = 0;
        for (auto& net : nets) {
            int leaf = net.router->route(ex.x);
            complexity += routerDepth(*net.router, ex.x) + includedLiterals(net.clauses[leaf]);
        }
        avgComplexity += complexity;
    }
    avgComplexity /= testData.size();
    cout << "XOR -- complexite de decision moyenne / prediction : " << avgComplexity << "\n";

    return 0;
}
