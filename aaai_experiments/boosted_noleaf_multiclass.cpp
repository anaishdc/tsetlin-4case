#include "../tsetlin_engine.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <chrono>

// Version multi-classe (one-vs-rest) de boosted_noleaf.cpp -- meme
// principe : boosting de mini-TM standards, sans routeur ni gel de
// features. Usage : ./boosted_noleaf_multiclass <iris|digits|mnist> [M] [C] [T]

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

inline void typeIFeedback(Clause& c, const vector<int>& x, double s, int clauseOut) {
    if (clauseOut == 0) {
        for (int i = 0; i < c.n; i++) { c.v[i].towardExclude(1.0/s); c.vbar[i].towardExclude(1.0/s); }
    } else {
        for (int i = 0; i < c.n; i++) {
            if (x[i] == 1) { c.v[i].towardInclude((s-1.0)/s); c.vbar[i].towardExclude(1.0/s); }
            else            { c.vbar[i].towardInclude((s-1.0)/s); c.v[i].towardExclude(1.0/s); }
        }
    }
}
inline void typeIIFeedback(Clause& c, const vector<int>& x, int clauseOut) {
    if (clauseOut == 1) {
        for (int i = 0; i < c.n; i++) {
            if (x[i] == 0) { if (!c.v[i].included())    c.v[i].towardInclude(1.0); }
            else           { if (!c.vbar[i].included()) c.vbar[i].towardInclude(1.0); }
        }
    }
}

struct MultiClauseRound {
    vector<Clause> clauses;
    vector<int> polarity;
    int T; double s;
    MultiClauseRound(int n, int N, int C, int T_, double s_) : T(T_), s(s_) {
        for (int k = 0; k < C; k++) { clauses.emplace_back(n, N); polarity.push_back(k % 2 == 0 ? +1 : -1); }
    }
    int rawSum(const vector<int>& x) const {
        int v = 0;
        for (size_t k = 0; k < clauses.size(); k++) v += polarity[k] * clauses[k].output(x);
        return v;
    }
    int output(const vector<int>& x) const { return rawSum(x) >= 0 ? 1 : 0; }
    double normalizedScore(const vector<int>& x) const {
        int v = max(-T, min(T, rawSum(x)));
        return (double)v / T;
    }
    void update(const vector<int>& x, int y) {
        int v = max(-T, min(T, rawSum(x)));
        for (size_t k = 0; k < clauses.size(); k++) {
            bool pos = polarity[k] > 0;
            int clauseOut = clauses[k].output(x);
            if (y == 1) {
                double p = (T - v) / (2.0 * T);
                if (!flip(p)) continue;
                if (pos) typeIFeedback(clauses[k], x, s, clauseOut); else typeIIFeedback(clauses[k], x, clauseOut);
            } else {
                double p = (T + v) / (2.0 * T);
                if (!flip(p)) continue;
                if (pos) typeIIFeedback(clauses[k], x, clauseOut); else typeIFeedback(clauses[k], x, s, clauseOut);
            }
        }
    }
};

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

void trainBoosted(const vector<LabeledExample>& trainBin, int n, int N, int C, int T, double s,
                  int total, int M, vector<MultiClauseRound>& rounds, vector<double>& alphas) {
    int Nex = (int)trainBin.size();
    vector<double> w(Nex, 1.0 / Nex);
    for (int m = 0; m < M; m++) {
        rounds.emplace_back(n, N, C, T, s);
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
    string which = argc > 1 ? argv[1] : "iris";
    int n, numClasses;
    vector<LabeledExample> trainRawAll, testRawAll;
    int N = 200, total = 100000, T = 15, C = 8; double s = 3.9;

    if (which == "iris") {
        n = 16; numClasses = 3;
        int M = argc > 2 ? atoi(argv[2]) : 5;
        C = argc > 3 ? atoi(argv[3]) : 8;
        T = argc > 4 ? atoi(argv[4]) : 15;
        int nbRuns = argc > 5 ? atoi(argv[5]) : 10;
        if (argc > 6) s = atof(argv[6]);
        if (argc > 7) N = atoi(argv[7]);
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
            vector<vector<MultiClauseRound>> ensembles(numClasses);
            vector<vector<double>> alphas(numClasses);
            for (int c = 0; c < numClasses; c++) {
                vector<LabeledExample> trainBin;
                for (auto& e : trainR) trainBin.push_back({e.x, e.y == c ? 1 : 0});
                trainBoosted(trainBin, n, N, C, T, s, total, M, ensembles[c], alphas[c]);
            }
            int correct = 0;
            for (auto& ex : testR) {
                vector<double> score(numClasses, 0);
                for (int c = 0; c < numClasses; c++)
                    for (size_t m = 0; m < ensembles[c].size(); m++)
                        score[c] += alphas[c][m] * ensembles[c][m].normalizedScore(ex.x);
                int pred = 0;
                for (int c = 1; c < numClasses; c++) if (score[c] > score[pred]) pred = c;
                if (pred == ex.y) correct++;
            }
            accs[run] = 100.0 * correct / testR.size();

            long totalLit = 0;
            for (int c = 0; c < numClasses; c++)
                for (auto& r : ensembles[c])
                    for (auto& cl : r.clauses)
                        for (int i = 0; i < cl.n; i++) { if (cl.v[i].included()) totalLit++; if (cl.vbar[i].included()) totalLit++; }
            cout << "  run " << run << " : acc=" << accs[run] << "%  complexite=" << totalLit << "\n" << flush;
        }
        double mean = 0; for (double a : accs) mean += a; mean /= nbRuns;
        double var = 0; for (double a : accs) var += (a-mean)*(a-mean); var /= nbRuns;
        cout << "iris (boosting mini-TM, sans routeur, " << nbRuns << " runs) : Mean=" << mean << " +/- " << sqrt(var) << "\n";
        return 0;
    } else if (which == "digits") {
        n = 192; numClasses = 10;
        trainRawAll = loadAll("../data/BinaryDigitsTrainingData192.txt", n);
        testRawAll  = loadAll("../data/BinaryDigitsTestData192.txt", n);
    } else { // mnist
        n = 784; numClasses = 10; N = 100;
        trainRawAll = loadAll("../data/MNISTTrainingOfficial60k.txt", n);
        testRawAll  = loadAll("../data/MNISTTestOfficial10k.txt", n);
    }
    int M = argc > 2 ? atoi(argv[2]) : 5;
    C = argc > 3 ? atoi(argv[3]) : 8;
    T = argc > 4 ? atoi(argv[4]) : 15;
    if (argc > 6) s = atof(argv[6]);
    if (argc > 7) N = atoi(argv[7]);

    cout << which << " (boosting mini-TM, sans routeur) : M=" << M << " C=" << C << " T=" << T
         << " s=" << s << " N=" << N << "\n" << flush;

    vector<vector<MultiClauseRound>> ensembles(numClasses);
    vector<vector<double>> alphas(numClasses);
    for (int c = 0; c < numClasses; c++) {
        vector<LabeledExample> trainBin;
        for (auto& e : trainRawAll) trainBin.push_back({e.x, e.y == c ? 1 : 0});
        auto t0 = chrono::steady_clock::now();
        trainBoosted(trainBin, n, N, C, T, s, total, M, ensembles[c], alphas[c]);
        auto t1 = chrono::steady_clock::now();
        cout << "  classe " << c << " terminee [" << chrono::duration<double>(t1-t0).count() << "s]\n" << flush;
    }

    int correct = 0;
    for (auto& ex : testRawAll) {
        vector<double> score(numClasses, 0);
        for (int c = 0; c < numClasses; c++)
            for (size_t m = 0; m < ensembles[c].size(); m++)
                score[c] += alphas[c][m] * ensembles[c][m].normalizedScore(ex.x);
        int pred = 0;
        for (int c = 1; c < numClasses; c++) if (score[c] > score[pred]) pred = c;
        if (pred == ex.y) correct++;
    }
    double acc = 100.0 * correct / testRawAll.size();

    long totalLit = 0;
    for (int c = 0; c < numClasses; c++)
        for (auto& net : ensembles[c])
            for (auto& cl : net.clauses)
                for (int i = 0; i < cl.n; i++) { if (cl.v[i].included()) totalLit++; if (cl.vbar[i].included()) totalLit++; }

    cout << which << " -- accuracy=" << acc << "%  complexite(litteraux totaux)=" << totalLit << "\n";
    return 0;
}
