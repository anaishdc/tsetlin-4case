#include "1_clause/tsetlin_engine_4case.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>


struct LabeledExample {
    vector<int> x;
    int y;
};

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

struct RandomStumpClause {
    Clause clause;
    int condFeature, condValue;
    vector<bool> active;
    RandomStumpClause(int n, int N, int f, int v) : clause(n, N), condFeature(f), condValue(v), active(n, true) {
        clause.v[f].state = 1; clause.vbar[f].state = 1;
        active[f] = false;  // gel 
    }
    bool applies(const vector<int>& x) const { return x[condFeature] == condValue; }
    bool isEmpty() const {
        for (int i = 0; i < clause.n; i++) if (clause.v[i].included() || clause.vbar[i].included()) return false;
        return true;
    }
};

int main(int argc, char** argv) {
    string which = argc > 1 ? argv[1] : "xor";
    if (which != "xor") {
        cerr << "Dataset inconnu : " << which << ". Seul \"xor\" est supporte.\n";
        return 1;
    }
    int n; vector<LabeledExample> trainRawAll, testRawAll;
    int N = 100, total = 1500, M = 100, nbRuns = 10;
    double S = 1, a = 0.3;

    n = 12;
    trainRawAll = loadDataset("data/NoisyXORTrainingData.txt", n);
    testRawAll  = loadDataset("data/NoisyXORTestData.txt", n);

    M = argc > 2 ? atoi(argv[2]) : 100;         // nombre total de clauses
    nbRuns = argc > 3 ? atoi(argv[3]) : 10;
    total = argc > 4 ? atoi(argv[4]) : 200 ;    // tirages d'entrainement PAR clause (sur son sous-ensemble)
    if (argc > 5) S = atof(argv[5]);
    if (argc > 6) a = atof(argv[6]);
    if (argc > 7) N = atoi(argv[7]);

    cout << which << " (stumps aleatoires, chaque clause = 1 round AdaBoost sequentiel, poids reportes d'un round au suivant) : M=" << M
         << " total/clause=" << total << " S=" << S << " a=" << a << " N=" << N << "\n" << flush;

    Params p = {S, a};
    uniform_int_distribution<int> featDist(0, n - 1);
    uniform_int_distribution<int> valDist(0, 1);

    vector<double> accs(nbRuns);

    for (int run = 0; run < nbRuns; run++) {
        rng.seed(150000 + run);
        int Nex = (int)trainRawAll.size();
        vector<double> w(Nex, 1.0 / Nex);
        vector<RandomStumpClause> clauses;
        vector<double> alphas;

        for (int m = 0; m < M; m++) {
            int f = featDist(rng), v = valDist(rng);
            clauses.emplace_back(n, N, f, v);

            // sous-ensemble d'indices ou la condition s'applique
            vector<int> subsetIdx;
            for (int i = 0; i < Nex; i++) if (clauses.back().applies(trainRawAll[i].x)) subsetIdx.push_back(i);
            if (subsetIdx.empty()) { alphas.push_back(0.0); continue; }

            vector<int> subPos, subNeg;
            for (int i : subsetIdx) (trainRawAll[i].y == 1 ? subPos : subNeg).push_back(i);
            vector<double> wPos, wNeg;
            for (int i : subPos) wPos.push_back(w[i]);
            for (int i : subNeg) wNeg.push_back(w[i]);
            discrete_distribution<int> posDist, negDist;
            if (!wPos.empty()) posDist = discrete_distribution<int>(wPos.begin(), wPos.end());
            if (!wNeg.empty()) negDist = discrete_distribution<int>(wNeg.begin(), wNeg.end());

            for (int t = 0; t < total; t++) {
                bool useNeg = subNeg.empty() ? false : (subPos.empty() ? true : !flip(0.5)); // Choisit d'équilibrer Positif / Négatif à 50%
                int idx = useNeg ? subNeg[negDist(rng)] : subPos[posDist(rng)];
                updateMasked(clauses.back().clause, trainRawAll[idx].x, trainRawAll[idx].y, p, clauses.back().active);
            }

            // Clause vide = abstention : ne doit ni compter dans l'erreur AdaBoost
            // ni modifier les poids, puisqu'elle sera de toute facon ignoree au vote final.
            if (clauses.back().isEmpty()) { alphas.push_back(0.0); continue; }

            double err = 0, wSum = 0;
            vector<bool> wrong(subsetIdx.size());
            for (size_t j = 0; j < subsetIdx.size(); j++) {
                int i = subsetIdx[j];
                wrong[j] = (clauses.back().clause.output(trainRawAll[i].x) != trainRawAll[i].y);
                if (wrong[j]) err += w[i];
                wSum += w[i];
            }
            constexpr double eps = 1e-6;
            err = max(eps, min(1.0 - eps, err / wSum));
            double alpha = 0.5 * log((1.0 - err) / err);
            alphas.push_back(alpha);

            double norm = 0;
            for (size_t j = 0; j < subsetIdx.size(); j++) {
                int i = subsetIdx[j];
                w[i] *= wrong[j] ? exp(alpha) : exp(-alpha);
            }
            for (int i = 0; i < Nex; i++) norm += w[i];
            for (int i = 0; i < Nex; i++) w[i] /= norm;
        }

        
        int correct = 0;
        int nKept = 0;
        for (auto& ex : testRawAll) {
            double score = 0;
            for (size_t m = 0; m < clauses.size(); m++) {
                if (!clauses[m].applies(ex.x) || clauses[m].isEmpty()) continue;
                int o = clauses[m].clause.output(ex.x);
                score += alphas[m] * (2.0 * o - 1.0);
            }
            int pred = score > 0 ? 1 : 0;
            if (pred == ex.y) correct++;
        }
        for (size_t m = 0; m < clauses.size(); m++) if (!clauses[m].isEmpty()) nKept++;
        accs[run] = 100.0 * correct / testRawAll.size();

        long totalLit = 0;
        for (auto& c : clauses) for (int i = 0; i < c.clause.n; i++) { if (c.clause.v[i].included()) totalLit++; if (c.clause.vbar[i].included()) totalLit++; }
        cout << "  run " << run << " : acc=" << accs[run] << "%  complexite=" << totalLit << "  clausesNonVides=" << nKept << "/" << clauses.size() << "\n" << flush;
    }
    double mean = 0; for (double a2 : accs) mean += a2; mean /= nbRuns;
    double var = 0; for (double a2 : accs) var += (a2-mean)*(a2-mean); var /= nbRuns;
    cout << which << " : Mean=" << mean << " +/- " << sqrt(var) << "\n";
    return 0;
}
