#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <functional>
#include <chrono>
using namespace std;

// -----------------------------------------------------------------
// Architecture alternative : pas d'arbre routeur du tout. Chaque
// "round" de boosting trouve directement (via les coefficients de
// Fourier ponderes) le sous-ensemble de variables le plus pertinent
// pour l'exemple pondere courant, puis construit une table de
// correspondance exacte sur ce sous-ensemble (vote majoritaire
// pondere par combinaison). Pas de gel de features, pas de
// "feuilles" qui deviennent inutiles a grand K -- chaque round est
// un mini-classifieur complet et independant sur SES variables
// choisies, combine par AdaBoost comme avant.
// -----------------------------------------------------------------

struct Example { vector<int> x; int y; };

vector<Example> loadAll(const string& path, int n) {
    vector<Example> data;
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

struct FourierLeaf {
    vector<int> subset;
    vector<double> table; // score pondere (proportion pondere de y=1) par combinaison
    int predict(const vector<int>& x) const {
        int idx = 0;
        for (int f : subset) idx = idx * 2 + x[f];
        return table[idx] >= 0.5 ? 1 : 0;
    }
    double score(const vector<int>& x) const {
        int idx = 0;
        for (int f : subset) idx = idx * 2 + x[f];
        return table[idx] - 0.5; // recentre autour de 0
    }
};

// Trouve le meilleur sous-ensemble (degre 1..maxDegree) par plus
// grand |coefficient de Fourier pondere|, construit la table de
// correspondance ponderee dessus.
FourierLeaf buildFourierLeaf(const vector<Example>& data, const vector<double>& w, int n, int maxDegree, int targetSize = -1) {
    int N = (int)data.size();
    vector<double> ys(N), wSum_(1, 0);
    double wTotal = 0;
    for (int i = 0; i < N; i++) { ys[i] = 2.0 * data[i].y - 1.0; wTotal += w[i]; }
    vector<vector<double>> xs(N, vector<double>(n));
    for (int i = 0; i < N; i++) for (int j = 0; j < n; j++) xs[i][j] = 2.0 * data[i].x[j] - 1.0;

    double bestAbs = -1; vector<int> bestSubset;

    auto evalSubset = [&](const vector<int>& subset) {
        double sum = 0;
        for (int i = 0; i < N; i++) {
            double prod = ys[i] * w[i];
            for (int j : subset) prod *= xs[i][j];
            sum += prod;
        }
        return sum / wTotal;
    };

    for (int deg = 1; deg <= maxDegree; deg++) {
        vector<int> idx(deg);
        // generation recursive des combinaisons
        function<void(int,int)> rec = [&](int start, int depth) {
            if (depth == deg) {
                double c = evalSubset(idx);
                if (fabs(c) > bestAbs) { bestAbs = fabs(c); bestSubset = idx; }
                return;
            }
            for (int f = start; f < n; f++) { idx[depth] = f; rec(f + 1, depth + 1); }
        };
        rec(0, 0);
        if (bestAbs > 0.5) break; // signal deja tres fort, pas la peine de chercher plus loin
    }

    // Extension gloutonne : ajoute une variable a la fois (cout O(n) par
    // etape au lieu de O(n^k) exhaustif) pour construire un sous-ensemble
    // bien plus grand -- imite un "conjonction a beaucoup de litteraux"
    // comme les clauses de l'ancien systeme, sans explosion combinatoire.
    if (targetSize > (int)bestSubset.size()) {
        vector<bool> used(n, false);
        for (int f : bestSubset) used[f] = true;
        vector<double> curProd(N);
        for (int i = 0; i < N; i++) {
            double p = ys[i] * w[i];
            for (int f : bestSubset) p *= xs[i][f];
            curProd[i] = p;
        }
        while ((int)bestSubset.size() < targetSize) {
            int bestF = -1; double bestGain = -1;
            for (int f = 0; f < n; f++) {
                if (used[f]) continue;
                double sum = 0;
                for (int i = 0; i < N; i++) sum += curProd[i] * xs[i][f];
                double c = fabs(sum / wTotal);
                if (c > bestGain) { bestGain = c; bestF = f; }
            }
            if (bestF == -1) break;
            used[bestF] = true;
            bestSubset.push_back(bestF);
            for (int i = 0; i < N; i++) curProd[i] *= xs[i][bestF];
        }
    }

    FourierLeaf leaf;
    leaf.subset = bestSubset;
    int K = 1 << bestSubset.size();
    vector<double> wSum(K, 0), wPos(K, 0);
    for (int i = 0; i < N; i++) {
        int idx = 0;
        for (int f : bestSubset) idx = idx * 2 + data[i].x[f];
        wSum[idx] += w[i];
        if (data[i].y == 1) wPos[idx] += w[i];
    }
    leaf.table.resize(K);
    for (int k = 0; k < K; k++) leaf.table[k] = wSum[k] > 0 ? wPos[k] / wSum[k] : 0.5;
    return leaf;
}

// Boosting AdaBoost, weak learner = FourierLeaf
void trainBoosted(const vector<Example>& trainBin, int n, int maxDegree, int M,
                  vector<FourierLeaf>& leaves, vector<double>& alphas, int targetSize = -1) {
    int Nex = (int)trainBin.size();
    vector<double> w(Nex, 1.0 / Nex);
    for (int m = 0; m < M; m++) {
        FourierLeaf leaf = buildFourierLeaf(trainBin, w, n, maxDegree, targetSize);
        leaves.push_back(leaf);

        double err = 0, wSum = 0;
        vector<bool> wrong(Nex);
        for (int i = 0; i < Nex; i++) {
            wrong[i] = (leaf.predict(trainBin[i].x) != trainBin[i].y);
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
    int n, numClasses, maxDegree, M;
    vector<Example> trainRawAll, testRawAll;
    int nbRuns = 1;

    if (which == "xor") {
        n = 12; numClasses = 2; maxDegree = 2; M = argc > 2 ? atoi(argv[2]) : 5;
        trainRawAll = loadAll("../data/NoisyXORTrainingData.txt", n);
        testRawAll  = loadAll("../data/NoisyXORTestData.txt", n);
    } else if (which == "iris") {
        n = 16; numClasses = 3; maxDegree = 2; M = argc > 2 ? atoi(argv[2]) : 5;
        nbRuns = argc > 4 ? atoi(argv[4]) : 40;
        vector<double> accs(nbRuns);
        auto allRawFull = loadAll("../data/BinaryIrisData.txt", n);
        if (argc > 3) maxDegree = atoi(argv[3]);
        for (int run = 0; run < nbRuns; run++) {
            mt19937 rng(20000 + run);
            auto allRaw = allRawFull;
            shuffle(allRaw.begin(), allRaw.end(), rng);
            int trainSize = (int)round(0.8 * allRaw.size());
            vector<Example> trainR(allRaw.begin(), allRaw.begin() + trainSize);
            vector<Example> testR(allRaw.begin() + trainSize, allRaw.end());

            vector<vector<FourierLeaf>> ensembles(numClasses);
            vector<vector<double>> alphas(numClasses);
            for (int c = 0; c < numClasses; c++) {
                vector<Example> trainBin;
                for (auto& e : trainR) trainBin.push_back({e.x, e.y == c ? 1 : 0});
                trainBoosted(trainBin, n, maxDegree, M, ensembles[c], alphas[c]);
            }
            int correct = 0;
            for (auto& ex : testR) {
                vector<double> score(numClasses, 0);
                for (int c = 0; c < numClasses; c++)
                    for (size_t m = 0; m < ensembles[c].size(); m++)
                        score[c] += alphas[c][m] * ensembles[c][m].score(ex.x);
                int pred = 0;
                for (int c = 1; c < numClasses; c++) if (score[c] > score[pred]) pred = c;
                if (pred == ex.y) correct++;
            }
            accs[run] = 100.0 * correct / testR.size();
        }
        double mean = 0; for (double a : accs) mean += a; mean /= nbRuns;
        double var = 0; for (double a : accs) var += (a-mean)*(a-mean); var /= nbRuns;
        cout << "iris (Fourier, " << nbRuns << " runs) : Mean=" << mean << " +/- " << sqrt(var) << "\n";
        return 0;
    } else if (which == "digits") {
        n = 192; numClasses = 10; maxDegree = 2; M = argc > 2 ? atoi(argv[2]) : 5;
        trainRawAll = loadAll("../data/BinaryDigitsTrainingData192.txt", n);
        testRawAll  = loadAll("../data/BinaryDigitsTestData192.txt", n);
    } else { // mnist
        n = 784; numClasses = 10; maxDegree = 1; M = argc > 2 ? atoi(argv[2]) : 8;
        trainRawAll = loadAll("../data/MNISTTrainingOfficial60k.txt", n);
        testRawAll  = loadAll("../data/MNISTTestOfficial10k.txt", n);
    }

    if (argc > 3) maxDegree = atoi(argv[3]);
    int targetSize = argc > 5 ? atoi(argv[5]) : -1;
    cout << which << " (Fourier, sans arbre) : maxDegree=" << maxDegree << " M=" << M
         << " targetSize=" << targetSize << "\n" << flush;

    vector<vector<FourierLeaf>> ensembles(numClasses);
    vector<vector<double>> alphas(numClasses);
    for (int c = 0; c < numClasses; c++) {
        vector<Example> trainBin;
        trainBin.reserve(trainRawAll.size());
        for (auto& e : trainRawAll) trainBin.push_back({e.x, e.y == c ? 1 : 0});
        auto tc0 = chrono::steady_clock::now();
        trainBoosted(trainBin, n, maxDegree, M, ensembles[c], alphas[c], targetSize);
        auto tc1 = chrono::steady_clock::now();
        cout << "  classe " << c << " terminee [" << chrono::duration<double>(tc1-tc0).count() << "s]\n" << flush;
    }

    int correct = 0;
    for (auto& ex : testRawAll) {
        vector<double> score(numClasses, 0);
        for (int c = 0; c < numClasses; c++)
            for (size_t m = 0; m < ensembles[c].size(); m++)
                score[c] += alphas[c][m] * ensembles[c][m].score(ex.x);
        int pred = 0;
        for (int c = 1; c < numClasses; c++) if (score[c] > score[pred]) pred = c;
        if (pred == ex.y) correct++;
    }
    double acc = 100.0 * correct / testRawAll.size();

    long totalComplexity = 0;
    for (int c = 0; c < numClasses; c++)
        for (auto& leaf : ensembles[c]) totalComplexity += leaf.subset.size();

    cout << which << " -- accuracy=" << acc << "%  complexite(nb variables totales)=" << totalComplexity << "\n";
    return 0;
}
