#include "../tsetlin_engine.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>

// =================================================================
// Multiplexeur booleen (benchmark classique Tsetlin Machine / XCS) :
// k bits d'adresse + 2^k bits de donnees, n = k + 2^k. Aucune feature
// seule n'est informative -- seule la combinaison adresse+donnee
// compte. Meme signature "interaction combinatoire pure" que le XOR
// bruite (validation_multiclause_xor_reel.cpp), mais sans bruit et
// sur un benchmark reconnu de la litterature.
// Usage : ./validation_multiplexer <6|11> [K] [nbRuns]
// =================================================================

vector<LabeledExample> loadDataset(const string& path, int nFeatures) {
    vector<LabeledExample> data;
    ifstream f(path);
    string line;
    while (getline(f, line)) {
        if (line.empty()) continue;
        istringstream iss(line);
        vector<int> x(nFeatures);
        for (int i = 0; i < nFeatures; i++) iss >> x[i];
        int y; iss >> y;
        data.push_back({ x, y });
    }
    return data;
}

double normalizedScore(const MultiClauseNetwork& net, const vector<int>& x) {
    int leaf = net.router->route(x);
    const Clause& c = net.clauses[leaf];
    int satisfied = 0, violated = 0, nLit = 0;
    for (int i = 0; i < c.n; i++) {
        if (c.v[i].included())    { nLit++; x[i] == 1 ? satisfied++ : violated++; }
        if (c.vbar[i].included()) { nLit++; x[i] == 0 ? satisfied++ : violated++; }
    }
    int rawScore = violated > 0 ? -violated : satisfied;
    return nLit > 0 ? (double)rawScore / nLit : 0.0;
}

vector<double> leafPuritiesWeighted(const RouterTree& tree, const vector<LabeledExample>& trainBin,
                                    const vector<double>& w) {
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
                       vector<MultiClauseNetwork>& nets, vector<double>& alphas,
                       vector<vector<double>>& purities) {
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

int main(int argc, char** argv) {
    const int k = argc > 1 ? atoi(argv[1]) : 3;   // nombre de bits informatifs (parite a k bits)
    const int n = 12;                             // nombre de features total (k informatifs + bruit)
    const int K = argc > 2 ? atoi(argv[2]) : 4;
    const int N = 500;
    const Params p = {1,0.3};
    const int total = 30000;
    const int discoverCount = argc > 4 ? atoi(argv[4]) : max(300, 3 * n);
    const double noiseTol = 0.40;  // 40% de bruit label (meme convention que XOR)
    const int M = 5;
    const int nbRuns = argc > 3 ? atoi(argv[3]) : 20;

    string trainPath = "../data/Parity" + to_string(k) + "TrainingData.txt";
    string testPath  = "../data/Parity" + to_string(k) + "TestData.txt";
    auto trainData = loadDataset(trainPath, n);
    auto testData  = loadDataset(testPath,  n);

    cout << fixed << setprecision(1);
    cout << "Dataset : Parity" << k << " train=" << trainData.size() << " test=" << testData.size()
         << "  K=" << K << " M=" << M << " (" << nbRuns << " runs)\n\n";

    vector<double> accsFull(nbRuns), accsBaseline(nbRuns);
    for (int run = 0; run < nbRuns; run++) {
        rng.seed(50000 + run);
        vector<MultiClauseNetwork> nets;
        vector<double> alphas;
        vector<vector<double>> purities;
        trainBoostedClass(trainData, n, K, N, p.S, p.a, total, discoverCount, noiseTol, M,
                          nets, alphas, purities);

        int correctFull = 0, correctBase = 0;
        for (auto& ex : testData) {
            double scoreFull = 0, scoreBase = 0;
            for (size_t m = 0; m < nets.size(); m++) {
                scoreFull += alphas[m] * normalizedScore(nets[m], ex.x);
                int leaf = nets[m].router->route(ex.x);
                scoreBase += alphas[m] * (purities[m][leaf] - 0.5);
            }
            int predFull = scoreFull > 0 ? 1 : 0;
            int predBase = scoreBase > 0 ? 1 : 0;
            if (predFull == ex.y) correctFull++;
            if (predBase == ex.y) correctBase++;
        }
        accsFull[run]     = 100.0 * correctFull / testData.size();
        accsBaseline[run] = 100.0 * correctBase / testData.size();
        cout << "  run " << run << " : systeme_complet=" << accsFull[run]
             << "%  baseline_routeurs_ponderes=" << accsBaseline[run] << "%\n" << flush;
    }

    auto stats = [&](vector<double> v, const string& label) {
        sort(v.begin(), v.end());
        double mean = 0; for (double a : v) mean += a; mean /= nbRuns;
        double var = 0; for (double a : v) var += (a - mean) * (a - mean); var /= nbRuns;
        int i5 = (int)(0.05 * nbRuns), i95 = (int)(0.95 * nbRuns);
        if (i95 >= nbRuns) i95 = nbRuns - 1;
        cout << label << " : Mean=" << mean << " +/- " << sqrt(var)
             << "  5%ile=" << v[i5] << "  95%ile=" << v[i95]
             << "  Min=" << v.front() << "  Max=" << v.back() << "\n";
        return mean;
    };

    cout << "\n";
    double meanFull = stats(accsFull, "Systeme complet (avec clauses)      ");
    double meanBase = stats(accsBaseline, "Baseline (routeurs ponderes, purete)");
    cout << "Ecart (apport reel des clauses)  : " << (meanFull - meanBase) << " points\n";

    return 0;
}
