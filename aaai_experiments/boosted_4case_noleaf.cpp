#include "../tsetlin_engine.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>

// -----------------------------------------------------------------
// Boosting SANS routeur ET avec notre regle a 4 cas (independante,
// pas Type I/II a seuil T partage). Chaque round de boosting = UNE
// seule clause (Clause::update standard, pas de masquage de feature,
// pas de coordination entre clauses) entrainee directement sur les
// donnees reponderees. La diversite vient uniquement du boosting
// (reponderation entre rounds), pas d'un mecanisme de coordination
// intra-round comme le seuil T de la vraie TM.
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

double normalizedScore(const Clause& c, const vector<int>& x) {
    int satisfied = 0, violated = 0, nLit = 0;
    for (int i = 0; i < c.n; i++) {
        if (c.v[i].included())    { nLit++; x[i] == 1 ? satisfied++ : violated++; }
        if (c.vbar[i].included()) { nLit++; x[i] == 0 ? satisfied++ : violated++; }
    }
    int rawScore = violated > 0 ? -violated : satisfied;
    return nLit > 0 ? (double)rawScore / nLit : 0.0;
}

// Un round = C clauses, chacune entrainee INDEPENDAMMENT (regle a 4
// cas standard, pas de coordination/seuil T) avec son propre tirage
// stochastique -- diversite type bagging, pas de mecanisme de la
// vraie TM. Sortie du round = vote majoritaire des C clauses.
struct ClauseGroup {
    vector<Clause> clauses;
    int output(const vector<int>& x) const {
        int votes = 0;
        for (auto& c : clauses) votes += c.output(x);
        return votes * 2 >= (int)clauses.size() ? 1 : 0;
    }
    double normalizedScore(const vector<int>& x) const {
        int votes = 0;
        for (auto& c : clauses) votes += c.output(x);
        return 2.0 * votes / clauses.size() - 1.0;
    }
};

void trainBoosted(const vector<LabeledExample>& trainBin, int n, int N, int C, double S, double a,
                  int total, int M, vector<ClauseGroup>& groups, vector<double>& alphas) {
    int Nex = (int)trainBin.size();
    vector<double> w(Nex, 1.0 / Nex);
    Params p = {S, S, S, S, a, a};
    for (int m = 0; m < M; m++) {
        groups.emplace_back();
        for (int k = 0; k < C; k++) {
            groups.back().clauses.emplace_back(n, N);
            WeightedSampler sampler(trainBin, w);   // reinstancie -> nouveau tirage stochastique independant
            for (int t = 0; t < total; t++) {
                const auto& ex = sampler.sample();
                groups.back().clauses.back().update(ex.x, ex.y, p);
            }
        }
        double err = 0, wSum = 0;
        vector<bool> wrong(Nex);
        for (int i = 0; i < Nex; i++) {
            wrong[i] = (groups.back().output(trainBin[i].x) != trainBin[i].y);
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
    int n; vector<LabeledExample> trainRawAll, testRawAll;
    int N = 100, total = 30000, M = 5, C = 4, nbRuns = 10;
    double S = 1, a = 0.3;

    if (which == "xor") {
        n = 12;
        trainRawAll = loadDataset("../data/NoisyXORTrainingData.txt", n);
        testRawAll  = loadDataset("../data/NoisyXORTestData.txt", n);
    } else if (which == "mux11") {
        n = 11;
        trainRawAll = loadDataset("../data/Mux11TrainingData.txt", n);
        testRawAll  = loadDataset("../data/Mux11TestData.txt", n);
    } else { // parity3
        n = 12;
        trainRawAll = loadDataset("../data/Parity3TrainingData.txt", n);
        testRawAll  = loadDataset("../data/Parity3TestData.txt", n);
    }
    M = argc > 2 ? atoi(argv[2]) : 5;
    C = argc > 3 ? atoi(argv[3]) : 4;
    nbRuns = argc > 4 ? atoi(argv[4]) : 10;
    total = argc > 5 ? atoi(argv[5]) : 30000;
    if (argc > 6) S = atof(argv[6]);
    if (argc > 7) a = atof(argv[7]);
    if (argc > 8) N = atoi(argv[8]);

    cout << which << " (boosting + regle a 4 cas, " << C << " clauses/round bagging, sans routeur) : M=" << M
         << " S=" << S << " a=" << a << " N=" << N << "\n" << flush;

    vector<double> accs(nbRuns);
    for (int run = 0; run < nbRuns; run++) {
        rng.seed(110000 + run);
        vector<ClauseGroup> groups;
        vector<double> alphas;
        trainBoosted(trainRawAll, n, N, C, S, a, total, M, groups, alphas);

        int correct = 0;
        for (auto& ex : testRawAll) {
            double score = 0;
            for (size_t m = 0; m < groups.size(); m++) score += alphas[m] * groups[m].normalizedScore(ex.x);
            int pred = score > 0 ? 1 : 0;
            if (pred == ex.y) correct++;
        }
        accs[run] = 100.0 * correct / testRawAll.size();

        long totalLit = 0;
        for (auto& g : groups) for (auto& c : g.clauses) for (int i = 0; i < c.n; i++) { if (c.v[i].included()) totalLit++; if (c.vbar[i].included()) totalLit++; }
        cout << "  run " << run << " : acc=" << accs[run] << "%  complexite=" << totalLit << "\n" << flush;
    }
    double mean = 0; for (double a2 : accs) mean += a2; mean /= nbRuns;
    double var = 0; for (double a2 : accs) var += (a2-mean)*(a2-mean); var /= nbRuns;
    cout << which << " : Mean=" << mean << " +/- " << sqrt(var) << "\n";
    return 0;
}
