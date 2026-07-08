#include "tsetlin_engine.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <iomanip>

// =================================================================
// Harness generique pour datasets de classification BINAIRE avec
// features deja binarisees (thermometre), un seul fichier (tous les
// exemples), split 80/20 reshuffle a chaque run -- meme
// methodologie que Iris/Breast Cancer.
//
// Usage : ./validation_generic_binary <chemin_data> <n_features> [K] [nbRuns]
// =================================================================

vector<LabeledExample> loadAll(const string& path, int nFeatures) {
    vector<LabeledExample> data;
    ifstream f(path);
    string line;
    while (getline(f, line)) {
        if (line.empty()) continue;
        istringstream iss(line);
        vector<int> x(nFeatures);
        for (int i = 0; i < nFeatures; i++) iss >> x[i];
        int label; iss >> label;
        data.push_back({ x, label });
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
        vector<WeightedLabeledExample> discoverBatch(dc, { vector<int>(n), 0, 0.0 });
        for (int i = 0; i < dc; i++) {
            discoverBatch[i].x = trainBin[idx[i]].x;
            discoverBatch[i].y = trainBin[idx[i]].y;
            discoverBatch[i].w = w[idx[i]];
        }

        auto router = buildWeightedRouterTree(discoverBatch, n, K, noiseTol);
        purities.push_back(leafPuritiesWeighted(*router, trainBin, w));
        nets.emplace_back(n, N, K, std::move(router), S, a);

        WeightedSampler sampler(trainBin, w);
        for (int t = 0; t < total; t++) {
            const auto& ex = sampler.sample();
            nets.back().update(ex.x, ex.y);
        }

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
    if (argc < 3) { cerr << "Usage: " << argv[0] << " <data_path> <n_features> [K] [nbRuns]\n"; return 1; }
    string dataPath = argv[1];
    const int    n             = atoi(argv[2]);
    const int    numClasses    = 2;
    const int    K             = argc > 3 ? atoi(argv[3]) : 2;
    const int    N             = 320;
    const Params p             = {1,1,1,1,0.3,0.3};
    const int    total         = 100000;
    const int    discoverCount = max(300, 3 * n);
    const double noiseTol      = 0.10;
    const int    M             = 16;
    const int    nbRuns        = argc > 4 ? atoi(argv[4]) : 20;

    auto allRaw = loadAll(dataPath, n);
    const int trainSize = (int)std::round(0.8 * allRaw.size());

    cout << fixed << setprecision(1);
    cout << "Dataset : " << dataPath << " -- " << allRaw.size() << " exemples  train=" << trainSize
         << " test=" << (allRaw.size() - trainSize) << "  n=" << n << " K=" << K << " M=" << M
         << " (" << nbRuns << " runs)\n\n";

    vector<double> accsFull(nbRuns), accsBaseline(nbRuns);
    for (int run = 0; run < nbRuns; run++) {
        rng.seed(20000 + run);

        vector<LabeledExample> shuffled = allRaw;
        shuffle(shuffled.begin(), shuffled.end(), rng);
        vector<LabeledExample> trainRaw(shuffled.begin(), shuffled.begin() + trainSize);
        vector<LabeledExample> testRaw(shuffled.begin() + trainSize, shuffled.end());

        vector<vector<MultiClauseNetwork>> ensembles(numClasses);
        vector<vector<double>> alphas(numClasses);
        vector<vector<vector<double>>> purities(numClasses);
        for (int c = 0; c < numClasses; c++) {
            vector<LabeledExample> trainBin;
            for (auto& e : trainRaw) trainBin.push_back({ e.x, e.y == c ? 1 : 0 });
            trainBoostedClass(trainBin, n, K, N, p.S1, p.a1, total, discoverCount, noiseTol, M,
                              ensembles[c], alphas[c], purities[c]);
        }

        int correctFull = 0, correctBase = 0;
        for (auto& e : testRaw) {
            vector<double> scoreFull(numClasses, 0);
            for (int c = 0; c < numClasses; c++)
                for (size_t m = 0; m < ensembles[c].size(); m++)
                    scoreFull[c] += alphas[c][m] * normalizedScore(ensembles[c][m], e.x);
            int predFull = 0;
            for (int c = 1; c < numClasses; c++) if (scoreFull[c] > scoreFull[predFull]) predFull = c;
            if (predFull == e.y) correctFull++;

            vector<double> scoreBase(numClasses, 0);
            for (int c = 0; c < numClasses; c++)
                for (size_t m = 0; m < ensembles[c].size(); m++) {
                    int leaf = ensembles[c][m].router->route(e.x);
                    scoreBase[c] += alphas[c][m] * purities[c][m][leaf];
                }
            int predBase = 0;
            for (int c = 1; c < numClasses; c++) if (scoreBase[c] > scoreBase[predBase]) predBase = c;
            if (predBase == e.y) correctBase++;
        }
        accsFull[run]     = 100.0 * correctFull / testRaw.size();
        accsBaseline[run] = 100.0 * correctBase / testRaw.size();
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
