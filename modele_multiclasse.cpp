#include "1_clause/tsetlin_engine_4case.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <thread>

// Exemple labellise : ([1,0,1,0,0], 0)
struct LabeledExample {
    vector<int> x;
    int y;
};

// Reprend la table de la regle a 4 cas, en sautant les features
// figees par le conditionnement (deja decidees, pas d'apprentissage).
inline void updateMasked(Clause& clause, const vector<int>& x, int y,
                         const Params& p, const vector<bool>& active) {
    double S = p.S, a = p.a;
    for (int i = 0; i < clause.n; i++) {
        if (!active[i]) continue;
        if (x[i] == 0) {
            if (y == 0) { clause.v[i].towardInclude(S); clause.vbar[i].towardExclude(S); }
            else        { clause.v[i].towardExclude(a); }
        } else {
            if (y == 0) { clause.v[i].towardExclude(S); clause.vbar[i].towardInclude(S); }
            else        { clause.vbar[i].towardExclude(a); }
        }
    }
}

// Version multi-classe (one-vs-rest) de random_stump_multifeature.cpp :
// chaque clause fixe K features aleatoires (au lieu d'une seule), notre
// regle a 4 cas (updateMasked), sans routeur, boosting AdaBoost independant
// par clause. Utilise pour Iris/Digits quand K=1 plafonne.

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

long g_pureCount = 0, g_mixedCount = 0, g_emptyCount = 0;

void trainOneClass(const vector<LabeledExample>& trainBin, int n, int N, int M, int total, int K,
                   double S, double a, vector<RandomStumpClause>& clauses, vector<double>& alphas) {
    Params p = {S, a};
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
        if (subsetIdx.empty()) { alphas.push_back(0.0); g_emptyCount++; continue; }

        vector<int> subPos, subNeg;
        for (int i : subsetIdx) (trainBin[i].y == 1 ? subPos : subNeg).push_back(i);
        if (subPos.empty() || subNeg.empty()) g_pureCount++; else g_mixedCount++;
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

        // Clause vide = abstention : ne doit ni compter dans l'erreur AdaBoost
        // ni modifier les poids, puisqu'elle sera de toute facon ignoree au vote final.
        if (clauses.back().isEmpty()) { alphas.push_back(0.0); continue; }

        double err = 0, wSum = 0;
        vector<bool> wrong(subsetIdx.size());
        for (size_t j = 0; j < subsetIdx.size(); j++) {
            int i = subsetIdx[j];
            wrong[j] = (clauses.back().clause.output(trainBin[i].x) != trainBin[i].y);
            if (wrong[j]) err += w[i];
            wSum += w[i];
        }
        constexpr double eps = 1e-6;
        err = max(eps, min(1.0 - eps, err / wSum));
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
    int N = 100, total = 20000, M = 100, nbRuns = 10, K = 2;
    double S = 1, a = 0.3;

    if (which == "iris") {
        n = 16; numClasses = 3;
        M = argc > 2 ? atoi(argv[2]) : 100;
        K = argc > 3 ? atoi(argv[3]) : 2;
        total = argc > 4 ? atoi(argv[4]) : 20000;
        nbRuns = argc > 5 ? atoi(argv[5]) : 10;
        if (argc > 6) S = atof(argv[6]);
        if (argc > 7) a = atof(argv[7]);
        if (argc > 8) N = atoi(argv[8]);

        cout << which << " (stumps a " << K << " features aleatoires, notre regle, sans routeur) : M=" << M
             << " total=" << total << " S=" << S << " a=" << a << " N=" << N << "\n" << flush;

        auto allRawFull = loadAll("data/BinaryIrisData.txt", n);
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
                trainOneClass(trainBin, n, N, M, total, K, S, a, ensembles[c], alphas[c]);
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
        long totalClauses = g_pureCount + g_mixedCount + g_emptyCount;
        cout << "  sous-ensembles : pur=" << g_pureCount << " (" << (100.0*g_pureCount/totalClauses) << "%)"
             << "  melange=" << g_mixedCount << " (" << (100.0*g_mixedCount/totalClauses) << "%)"
             << "  vide=" << g_emptyCount << " (" << (100.0*g_emptyCount/totalClauses) << "%)\n";
        return 0;
    }

    // digits ou mnist (meme structure, seule la source de donnees et la taille changent)
    if (which == "digits") {
        n = 192; numClasses = 10;
        M = argc > 2 ? atoi(argv[2]) : 100;
        K = argc > 3 ? atoi(argv[3]) : 2;
        total = argc > 4 ? atoi(argv[4]) : 20000;
        nbRuns = argc > 5 ? atoi(argv[5]) : 5;
        if (argc > 6) S = atof(argv[6]);
        if (argc > 7) a = atof(argv[7]);
        if (argc > 8) N = atoi(argv[8]);
    } else { // mnist
        n = 784; numClasses = 10;
        N = 256;
        M = argc > 2 ? atoi(argv[2]) : 300;
        K = argc > 3 ? atoi(argv[3]) : 1;
        total = argc > 4 ? atoi(argv[4]) : 20000;
        nbRuns = argc > 5 ? atoi(argv[5]) : 1;
        if (argc > 6) S = atof(argv[6]);
        if (argc > 7) a = atof(argv[7]);
        if (argc > 8) N = atoi(argv[8]);
    }

    cout << which << " (stumps a " << K << " features aleatoires, notre regle, sans routeur, " << nbRuns << " runs) : M=" << M
         << " total=" << total << " S=" << S << " a=" << a << " N=" << N << "\n" << flush;

    // Digits : data/BinaryDigitsTrainingData192.txt (repo). MNIST : cloner
    // https://github.com/cair/fast-tsetlin-machine-with-mnist-demo.git puis
    // dezipper BinarizedMNISTData.zip (MNISTTraining.txt / MNISTTest.txt).
    auto trainRawAll = which == "digits"
        ? loadAll("data/BinaryDigitsTrainingData192.txt", n)
        : loadAll("fast-tsetlin-machine-with-mnist-demo/MNISTTraining.txt", n);
    auto testRawAll = which == "digits"
        ? loadAll("data/BinaryDigitsTestData192.txt", n)
        : loadAll("fast-tsetlin-machine-with-mnist-demo/MNISTTest.txt", n);
    cout << "train=" << trainRawAll.size() << " test=" << testRawAll.size() << "\n" << flush;

    unsigned nThreads = thread::hardware_concurrency();
    if (nThreads == 0) nThreads = 4;
    cout << "threads disponibles : " << nThreads << "\n" << flush;

    vector<double> accs(nbRuns);
    for (int run = 0; run < nbRuns; run++) {
        vector<vector<RandomStumpClause>> ensembles(numClasses);
        vector<vector<double>> alphas(numClasses);
        // une classe = un thread independant (rng thread_local, seede
        // distinctement) ; les classes sont totalement independantes
        // (one-vs-rest), aucun changement a l'algorithme lui-meme.
        auto t0 = chrono::steady_clock::now();
        vector<thread> workers;
        for (int c = 0; c < numClasses; c++) {
            workers.emplace_back([&, c]() {
                rng.seed(20000 + run * 1000 + c);
                vector<LabeledExample> trainBin;
                trainBin.reserve(trainRawAll.size());
                for (auto& e : trainRawAll) trainBin.push_back({e.x, e.y == c ? 1 : 0});
                trainOneClass(trainBin, n, N, M, total, K, S, a, ensembles[c], alphas[c]);
            });
        }
        for (auto& th : workers) th.join();
        auto t1 = chrono::steady_clock::now();
        cout << "  entrainement run " << run << " (parallele) [" << chrono::duration<double>(t1-t0).count() << "s]\n" << flush;

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
