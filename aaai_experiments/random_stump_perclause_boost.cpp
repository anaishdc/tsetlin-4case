#include "../tsetlin_engine.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>

// -----------------------------------------------------------------
// Chaque clause-stump (une seule condition aleatoire fixe, notre
// regle a 4 cas) est son PROPRE round de boosting independant --
// pas de regroupement C-par-round avec vote a egalite. L'erreur/alpha
// de chaque clause est calculee UNIQUEMENT sur le sous-ensemble ou
// elle s'applique (elle n'a pas d'avis ailleurs). Les poids des
// exemples HORS de son sous-ensemble restent inchanges pour ce round.
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

struct RandomStumpClause {
    Clause clause;
    int condFeature, condValue;
    vector<bool> active;
    RandomStumpClause(int n, int N, int f, int v) : clause(n, N), condFeature(f), condValue(v), active(n, true) {
        clause.v[f].state = 1; clause.vbar[f].state = 1;
        active[f] = false;  // gel REEL : exclu de l'apprentissage, pas seulement de l'etat initial
    }
    bool applies(const vector<int>& x) const { return x[condFeature] == condValue; }
    bool isEmpty() const {
        for (int i = 0; i < clause.n; i++) if (clause.v[i].included() || clause.vbar[i].included()) return false;
        return true;
    }
};

int main(int argc, char** argv) {
    string which = argc > 1 ? argv[1] : "xor";
    int n; vector<LabeledExample> trainRawAll, testRawAll;
    int N = 100, total = 1500, M = 100, nbRuns = 10;
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
    M = argc > 2 ? atoi(argv[2]) : 100;         // nombre TOTAL de clauses-stump (chacune son propre round)
    nbRuns = argc > 3 ? atoi(argv[3]) : 10;
    total = argc > 4 ? atoi(argv[4]) : 1500;    // tirages d'entrainement PAR clause (sur son sous-ensemble)
    if (argc > 5) S = atof(argv[5]);
    if (argc > 6) a = atof(argv[6]);
    if (argc > 7) N = atoi(argv[7]);

    cout << which << " (stumps aleatoires, CHAQUE clause = 1 round AdaBoost independant) : M=" << M
         << " total/clause=" << total << " S=" << S << " a=" << a << " N=" << N << "\n" << flush;

    Params p = {S, S, S, S, a, a};
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
                bool useNeg = subNeg.empty() ? false : (subPos.empty() ? true : !flip(0.5));
                int idx = useNeg ? subNeg[negDist(rng)] : subPos[posDist(rng)];
                updateMasked(clauses.back().clause, trainRawAll[idx].x, trainRawAll[idx].y, p, clauses.back().active);
            }

            double err = 0, wSum = 0;
            vector<bool> wrong(subsetIdx.size());
            for (size_t j = 0; j < subsetIdx.size(); j++) {
                int i = subsetIdx[j];
                wrong[j] = (clauses.back().clause.output(trainRawAll[i].x) != trainRawAll[i].y);
                if (wrong[j]) err += w[i];
                wSum += w[i];
            }
            err = max(1e-6, min(0.999, err / wSum));
            double alpha = 0.5 * log((1.0 - err) / err);
            alphas.push_back(alpha);
            if (m < 15) cerr << "    [diag] clause " << m << " : cond=(x" << f << "==" << v << ")"
                             << " subsetSize=" << subsetIdx.size() << " err=" << err << " alpha=" << alpha
                             << " empty=" << clauses.back().isEmpty() << "\n";
            double norm = 0;
            for (size_t j = 0; j < subsetIdx.size(); j++) {
                int i = subsetIdx[j];
                w[i] *= wrong[j] ? exp(alpha) : exp(-alpha);
            }
            for (int i = 0; i < Nex; i++) norm += w[i];
            for (int i = 0; i < Nex; i++) w[i] /= norm;
        }

        int correct = 0;
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
        accs[run] = 100.0 * correct / testRawAll.size();

        long totalLit = 0;
        for (auto& c : clauses) for (int i = 0; i < c.clause.n; i++) { if (c.clause.v[i].included()) totalLit++; if (c.clause.vbar[i].included()) totalLit++; }
        cout << "  run " << run << " : acc=" << accs[run] << "%  complexite=" << totalLit << "\n" << flush;
    }
    double mean = 0; for (double a2 : accs) mean += a2; mean /= nbRuns;
    double var = 0; for (double a2 : accs) var += (a2-mean)*(a2-mean); var /= nbRuns;
    cout << which << " : Mean=" << mean << " +/- " << sqrt(var) << "\n";
    return 0;
}
