#include "../tsetlin_engine.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>

// -----------------------------------------------------------------
// Algo general : chaque clause-stump construit PROGRESSIVEMENT sa
// condition (1 feature aleatoire a la fois), et decide de la garder
// via un chi2 pondere calcule LOCALEMENT (dans le sous-ensemble deja
// filtre par les conditions precedentes), pas globalement sur tout le
// dataset. Ca resout 2 problemes en meme temps :
//  - le chi2 global aurait le meme angle mort que Fourier degre 1 sur
//    XOR (chaque feature seule a une correlation nulle) ; mais une
//    fois conditionne sur une premiere feature, le chi2 local de la
//    seconde devient fort (structure XOR devient lineaire)
//  - le chi2 tient compte de la taille de l'echantillon (contrairement
//    a la correlation brute), donc pas de sur-ajustement sur des
//    sous-ensembles minuscules (probleme qui avait fait echouer la
//    version biaisee-Fourier sur Iris)
// A chaque etape on ne regarde que quelques candidats aleatoires
// (maxTries), pas toutes les features -- donc ce n'est PAS un routeur
// exhaustif/adaptatif, juste une petite loupe locale. K est decouvert
// automatiquement (pas fixe a la main), jusqu'a maxK.
// Le reste (regle a 4 cas + updateMasked + boosting AdaBoost par
// clause independante) est identique a random_stump_perclause_boost.cpp.
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

struct ChiStumpClause {
    Clause clause;
    vector<int> condFeatures, condValues;
    vector<bool> active;
    ChiStumpClause(int n, int N, const vector<int>& feats, const vector<int>& vals)
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

struct CandRes { double chi2; int value; };

CandRes evalFeature(const vector<LabeledExample>& data, const vector<int>& subset,
                     const vector<double>& w, int f, double scale) {
    // 'w' sont des poids AdaBoost normalises (somme=1) : il faut les remettre
    // a l'echelle "nombre d'exemples effectif" (scale = Nex) avant de calculer
    // le chi2, sinon la statistique est ecrasee par un facteur ~Nex et ne
    // depasse jamais aucun seuil de significativite.
    double w00 = 0, w01 = 0, w10 = 0, w11 = 0;
    for (int i : subset) {
        double wi = w[i] * scale;
        int fv = data[i].x[f], y = data[i].y;
        if (fv == 0) { if (y == 0) w00 += wi; else w01 += wi; }
        else         { if (y == 0) w10 += wi; else w11 += wi; }
    }
    double N = w00 + w01 + w10 + w11;
    if (N <= 1e-12) return {0.0, 0};
    double row0 = w00 + w01, row1 = w10 + w11, col0 = w00 + w10, col1 = w01 + w11;
    double obs[2][2] = {{w00, w01}, {w10, w11}};
    double rows[2] = {row0, row1}, cols[2] = {col0, col1};
    double chi2 = 0;
    for (int r = 0; r < 2; r++) for (int c = 0; c < 2; c++) {
        double exp = rows[r] * cols[c] / N;
        if (exp > 1e-9) chi2 += (obs[r][c] - exp) * (obs[r][c] - exp) / exp;
    }
    double pur0 = row0 > 1e-12 ? max(w00, w01) / row0 : 0;
    double pur1 = row1 > 1e-12 ? max(w10, w11) / row1 : 0;
    int value = pur1 >= pur0 ? 1 : 0;
    return {chi2, value};
}

// Une parite/XOR de degre d a une correlation marginale NULLE a chaque
// niveau tant qu'il reste >=2 variables pertinentes non fixees -- le chi2
// ne peut donc jamais choisir QUELLE feature ajouter au conditionnement.
// En revanche il sait tres bien dire QUAND s'arreter : des qu'une seule
// variable explique fortement le sous-ensemble courant (corr locale
// significative), c'est exactement le cas que notre regle a 4 cas sait
// apprendre comme un litteral -- on s'arrete alors et on NE fige PAS
// cette derniere feature (sinon le clause devient pure -> vide -> abstient).
// Tant qu'aucun signal n'est visible (structure encore symetrique/cachee),
// on avance a l'aveugle : on fige UNE feature aleatoire et on continue.
void growCondition(const vector<LabeledExample>& data, const vector<double>& w, int n,
                    int maxK, int minSubsetSize, double chi2Threshold, int maxTries,
                    vector<int>& feats, vector<int>& vals) {
    vector<int> subset(data.size());
    iota(subset.begin(), subset.end(), 0);
    vector<bool> used(n, false);
    uniform_int_distribution<int> valDist(0, 1);
    double scale = (double)data.size(); // remet les poids AdaBoost normalises a l'echelle "effectifs"
    for (int k = 0; k < maxK; k++) {
        vector<int> pool;
        for (int f = 0; f < n; f++) if (!used[f]) pool.push_back(f);
        if (pool.empty()) break;
        shuffle(pool.begin(), pool.end(), rng);
        int nTry = min((int)pool.size(), maxTries);

        int bestFeat = -1, bestVal = 0; double bestChi2 = -1;
        for (int tries = 0; tries < nTry; tries++) {
            int f = pool[tries];
            auto res = evalFeature(data, subset, w, f, scale);
            if (res.chi2 > bestChi2) { bestChi2 = res.chi2; bestFeat = f; bestVal = res.value; }
        }
        if (getenv("CHI2_DEBUG")) cerr << "    [dbg] k=" << k << " bestFeat(signal)=" << bestFeat << " chi2=" << bestChi2 << " subsetSize=" << subset.size() << "\n";

        if (bestFeat != -1 && bestChi2 >= chi2Threshold) {
            // signal resolvant trouve -> on s'arrete ICI, on laisse cette
            // feature au litteral-learning normal (pas de gel).
            break;
        }
        // aucun signal -> avance a l'aveugle (comme la version stump aleatoire)
        int f = pool[0];
        int v = valDist(rng);
        vector<int> newSubset;
        for (int i : subset) if (data[i].x[f] == v) newSubset.push_back(i);
        if ((int)newSubset.size() < minSubsetSize) break;
        feats.push_back(f); vals.push_back(v);
        used[f] = true;
        subset = newSubset;
    }
}

int main(int argc, char** argv) {
    string which = argc > 1 ? argv[1] : "xor";
    int n; vector<LabeledExample> trainRawAll, testRawAll;
    int N = 100, total = 1500, M = 100, nbRuns = 10;
    double S = 1, a = 0.3;
    int maxK = 4, maxTries = 6;
    double chi2Threshold = 3.84;

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
    M = argc > 2 ? atoi(argv[2]) : 100;
    nbRuns = argc > 3 ? atoi(argv[3]) : 10;
    total = argc > 4 ? atoi(argv[4]) : 1500;
    if (argc > 5) S = atof(argv[5]);
    if (argc > 6) a = atof(argv[6]);
    if (argc > 7) N = atoi(argv[7]);
    if (argc > 8) maxK = atoi(argv[8]);
    int minSubsetSize = argc > 9 ? atoi(argv[9]) : max(10, (int)trainRawAll.size() / 20);
    if (argc > 10) maxTries = atoi(argv[10]);
    if (argc > 11) chi2Threshold = atof(argv[11]);

    cout << which << " (stumps chi2 progressifs, K auto, notre regle, sans routeur) : M=" << M
         << " total/clause=" << total << " S=" << S << " a=" << a << " N=" << N
         << " maxK=" << maxK << " minSubset=" << minSubsetSize << " maxTries=" << maxTries
         << " chi2Thr=" << chi2Threshold << "\n" << flush;

    Params p = {S, S, S, S, a, a};

    vector<double> accs(nbRuns);
    for (int run = 0; run < nbRuns; run++) {
        rng.seed(170000 + run);
        int Nex = (int)trainRawAll.size();
        vector<double> w(Nex, 1.0 / Nex);
        vector<ChiStumpClause> clauses;
        vector<double> alphas;
        long kSum = 0;

        for (int m = 0; m < M; m++) {
            vector<int> feats, vals;
            growCondition(trainRawAll, w, n, maxK, minSubsetSize, chi2Threshold, maxTries, feats, vals);
            kSum += feats.size();
            clauses.emplace_back(n, N, feats, vals);

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
            for (size_t j = 0; j < subsetIdx.size(); j++) {
                int i = subsetIdx[j];
                w[i] *= wrong[j] ? exp(alpha) : exp(-alpha);
            }
            double norm = 0;
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
        cout << "  run " << run << " : acc=" << accs[run] << "%  K_moyen=" << (double)kSum / M << "\n" << flush;
    }
    double mean = 0; for (double a2 : accs) mean += a2; mean /= nbRuns;
    double var = 0; for (double a2 : accs) var += (a2-mean)*(a2-mean); var /= nbRuns;
    cout << which << " : Mean=" << mean << " +/- " << sqrt(var) << "\n";
    return 0;
}
