#include "../tsetlin_engine.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <thread>

// MNIST avec la methode validee : stumps aleatoires (K features fixees),
// notre regle a 4 cas (updateMasked), sans routeur, chaque clause = un
// round AdaBoost independant. Meme structure que random_stump_multiclass.cpp
// (Digits), adaptee a n=784.

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

void trainOneClass(const vector<LabeledExample>& trainBin, int n, int N, int M, int total, int K,
                   double S, double a, vector<RandomStumpClause>& clauses, vector<double>& alphas) {
    Params p = {S, S, S, S, a, a};
    uniform_int_distribution<int> valDist(0, 1);
    int Nex = (int)trainBin.size();
    vector<double> w(Nex, 1.0 / Nex);

    for (int m = 0; m < M; m++) {
        vector<int> pool(n); for (int i = 0; i < n; i++) pool[i] = i;
        shuffle(pool.begin(), pool.end(), rng);
        vector<int> feats, vals;
        for (int k = 0; k < K; k++) { feats.push_back(pool[k]); vals.push_back(valDist(rng)); }
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
    int n = 784, numClasses = 10;
    int N = 256, total = 5000, M = 200, nbRuns = 1, K = 1;
    double S = 1, a = 0.3;

    M = argc > 1 ? atoi(argv[1]) : 200;
    total = argc > 2 ? atoi(argv[2]) : 5000;
    nbRuns = argc > 3 ? atoi(argv[3]) : 1;
    K = argc > 4 ? atoi(argv[4]) : 1;
    if (argc > 5) S = atof(argv[5]);
    if (argc > 6) a = atof(argv[6]);
    if (argc > 7) N = atoi(argv[7]);

    cout << "mnist (stumps aleatoires + notre regle, sans routeur, " << nbRuns << " runs) : M=" << M
         << "/classe total=" << total << " K=" << K << " S=" << S << " a=" << a << " N=" << N << "\n" << flush;

    auto trainRawAll = loadAll("../data/MNISTTrainingOfficial60k.txt", n);
    auto testRawAll  = loadAll("../data/MNISTTestOfficial10k.txt", n);
    cout << "train=" << trainRawAll.size() << " test=" << testRawAll.size() << "\n" << flush;

    unsigned nThreads = thread::hardware_concurrency();
    if (nThreads == 0) nThreads = 4;
    cout << "threads disponibles : " << nThreads << "\n" << flush;

    vector<double> accs(nbRuns);
    for (int run = 0; run < nbRuns; run++) {
        vector<vector<RandomStumpClause>> ensembles(numClasses);
        vector<vector<double>> alphas(numClasses);
        // une classe = un thread independant (chacune a son propre rng
        // thread_local, seede distinctement) ; les 10 classes sont
        // totalement independantes (one-vs-rest), donc parallelisables
        // sans aucun changement a l'algorithme lui-meme.
        auto t0 = chrono::steady_clock::now();
        vector<thread> workers;
        for (int c = 0; c < numClasses; c++) {
            workers.emplace_back([&, c]() {
                rng.seed(20000 + run * 1000 + c);
                vector<LabeledExample> trainBin;
                trainBin.reserve(trainRawAll.size());
                for (auto& e : trainRawAll) trainBin.push_back({e.x, e.y == c ? 1 : 0});
                auto tc0 = chrono::steady_clock::now();
                trainOneClass(trainBin, n, N, M, total, K, S, a, ensembles[c], alphas[c]);
                auto tc1 = chrono::steady_clock::now();
                cout << "  classe " << c << " terminee [" << chrono::duration<double>(tc1-tc0).count() << "s]\n" << flush;
            });
        }
        for (auto& th : workers) th.join();
        auto t1 = chrono::steady_clock::now();
        cout << "  entrainement total (parallele) [" << chrono::duration<double>(t1-t0).count() << "s]\n" << flush;
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
    cout << "mnist : Mean=" << mean << " +/- " << sqrt(var) << "\n";
    return 0;
}
