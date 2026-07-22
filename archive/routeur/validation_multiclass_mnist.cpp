#include "../tsetlin_engine.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <chrono>

// DIAGNOSTIC TEMPORAIRE : lit la memoire residente reelle du processus
// (VmRSS) directement depuis /proc, pour voir OU la memoire croit vraiment
// plutot que de deviner a partir d'un calcul theorique.
long vmRssKB() {
    ifstream f("/proc/self/status");
    string line;
    while (getline(f, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            long kb; sscanf(line.c_str(), "VmRSS: %ld kB", &kb);
            return kb;
        }
    }
    return -1;
}

// =================================================================
// Validation multi-classe MNIST -- meme methodologie que Iris/
// Digits/XOR : boosting AdaBoost avec routeur PONDERE reconstruit a
// chaque round (tsetlin_engine.h), combineur par
// argmax sur la somme ponderee (alpha) des scores normalises, et
// baseline "routeurs ponderes seuls" (purete, sans clause) pour
// verifier que les clauses apportent bien quelque chose.
//
// Papier (Section 5.5) : binarisation par seuillage a 0.3 (pixel >
// 0.3 -> 1, sinon 0), 60000 train / 10000 test, reproductible via
// https://github.com/cair/fast-tsetlin-machine-with-mnist-demo
// (BinarizedMNISTData.zip). Donnees OFFICIELLES recuperees de ce
// depot (git clone + unzip) -- data/MNISTTrainingOfficial60k.txt /
// MNISTTestOfficial10k.txt, 60000/10000 exemples, deja binarisees
// au seuil 0.3, labels verifies conformes a la distribution MNIST
// standard (5923 zeros, 6742 uns, etc.). "total" (exemples tires
// AVEC REMISE par round de boosting) ne depend pas de la taille du
// pool source, donc passer de 6000 a 60000 exemples train ne change
// quasiment pas le temps d'entrainement -- seule l'evaluation sur
// 10000 (au lieu de 1000) exemples test grandit, mais cela reste
// negligeable face au cout d'entrainement.
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

// Purete ponderee d'une feuille (pour la baseline "routeur seul")
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

// Sampler pondere : pool positif/negatif choisis 50/50, exemple DANS
// le pool tire proportionnellement a son poids de boosting.
struct WeightedSampler {
    const vector<LabeledExample>& data;
    vector<int> posIdx, negIdx;
    discrete_distribution<int> posDist, negDist;
    // Les indices par classe (posIdx/negIdx) ne dependent que de y, jamais de w :
    // calcules une seule fois ici, reutilises a chaque round (economise une
    // reallocation complete par round -- important quand data est grand, ex. MNIST).
    WeightedSampler(const vector<LabeledExample>& d) : data(d) {
        for (int i = 0; i < (int)d.size(); i++) (d[i].y == 1 ? posIdx : negIdx).push_back(i);
    }
    // A rappeler a chaque round : ne reconstruit que les tables de tirage
    // pondere (posDist/negDist), pas les listes d'indices.
    void refreshWeights(const vector<double>& w) {
        vector<double> wPos, wNeg;
        wPos.reserve(posIdx.size()); wNeg.reserve(negIdx.size());
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

// Entraine un ensemble boosted (M rounds) pour UNE classe (probleme
// un-contre-tous) : routeur ET clauses ponderes a chaque round.
void trainBoostedClass(const vector<LabeledExample>& trainBin, int n, int K, int N,
                       double S, double a, int total, int discoverCount, double noiseTol, int M,
                       vector<MultiClauseNetwork>& nets, vector<double>& alphas,
                       vector<vector<double>>& purities) {
    int Nex = (int)trainBin.size();
    vector<double> w(Nex, 1.0 / Nex);

    // Buffers reutilises d'un round a l'autre (memes tailles a chaque fois,
    // donc aucune reallocation apres le premier round) -- evite de refaire
    // ~discoverCount copies de x (784 entiers chacune sur MNIST) M fois de
    // suite, qui fragmentait le tas et gonflait la memoire residente.
    vector<int> idx(Nex);
    int dc = min(Nex, discoverCount);
    vector<WeightedLabeledExample> discoverBatch(dc, { vector<int>(n), 0, 0.0 });
    WeightedSampler sampler(trainBin);

    for (int m = 0; m < M; m++) {
        for (int i = 0; i < Nex; i++) idx[i] = i;
        sort(idx.begin(), idx.end(), [&](int a2, int b2) { return w[a2] > w[b2]; });
        for (int i = 0; i < dc; i++) {
            discoverBatch[i].x = trainBin[idx[i]].x;   // meme taille (n) => pas de reallocation
            discoverBatch[i].y = trainBin[idx[i]].y;
            discoverBatch[i].w = w[idx[i]];
        }

        auto router = buildWeightedRouterTree(discoverBatch, n, K, noiseTol);
        purities.push_back(leafPuritiesWeighted(*router, trainBin, w));
        nets.emplace_back(n, N, K, std::move(router), S, a);

        sampler.refreshWeights(w);
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
    const int    n             = 784;
    const int    numClasses    = 10;
    const int    K             = argc > 1 ? atoi(argv[1]) : 64;   // K surchargeable : argv[1]. Historique complet (voir tsetlin_engine.h
                                        // pour le detail des bugs de routeur trouves et corriges) :
                                        // (1) K=2 choisi initialement via un diagnostic non-boosted
                                        // non representatif -- mauvais choix ; (2) bug de lookahead
                                        // manquant (routeur choisissait une feature de bruit au hasard
                                        // sur structure type parite, confirme via Noisy XOR : 48.9% vs
                                        // 100%) -- corrige en restaurant un lookahead 1 niveau,
                                        // applique de facon identique a TOUS les datasets (aucune
                                        // branche/seuil dependant de n) ; (3) cout du lookahead
                                        // exhaustif O(n^2) impraticable a n=784 (>1000s/config) --
                                        // resolu en limitant le lookahead aux LOOKAHEAD_TOPM=30
                                        // meilleurs candidats gloutons (meme algorithme, meme
                                        // constante, pour tous les datasets) ; (4) gain de split
                                        // pondere par la masse de la feuille (evite qu'une feuille
                                        // "fourre-tout" ne soit jamais fragmentee). Une fois ces
                                        // quatre corrections en place, K-sweep refait proprement sur
                                        // donnees officielles (M=16, meme algorithme partout) :
                                        //   K=16->84.5%  K=32->89.3%  K=64->92.3% (retenu)
                                        // Tendance croissante avec K, pas de plateau observe jusqu'a
                                        // K=64 (52 min/run) -- diagnostic confirme (voir session) que
                                        // la limite residuelle est la capacite representationnelle
                                        // d'une seule clause conjonctive par feuille (memes feuilles
                                        // bien partitionnees, err->chance sur les rounds de boosting
                                        // tardifs), pas le routeur.
    const int    N             = 320;
    const Params p             = {1,0.3};
    const int    total         = 100000;
    const int    discoverCount = max(300, 3 * n);   // regle generale : proportionnel a n, pas a la taille du dataset
    const double noiseTol      = 0.05;
    const int    M             = 16;    // rounds de boosting (coherence Iris/Digits/XOR)
    const int    nbRuns        = 1;    // ~13 min/run mesure -- 1 seul run pour rester rapide

    auto trainRawAll = loadAll("../data/MNISTTrainingOfficial60k.txt", n);
    auto testRawAll  = loadAll("../data/MNISTTestOfficial10k.txt",     n);
    cerr << "[RSS] apres chargement train+test : " << vmRssKB()/1024.0 << " Mo\n" << flush;

    cout << fixed << setprecision(1);
    cout << "Dataset : train=" << trainRawAll.size() << " test=" << testRawAll.size()
         << "  K=" << K << " M=" << M << " (" << nbRuns << " runs)\n\n";

    vector<double> accsFull(nbRuns), accsBaseline(nbRuns);
    for (int run = 0; run < nbRuns; run++) {
        auto t0 = chrono::steady_clock::now();
        rng.seed(23000 + run);

        vector<vector<MultiClauseNetwork>> ensembles(numClasses);
        vector<vector<double>> alphas(numClasses);
        vector<vector<vector<double>>> purities(numClasses);
        for (int c = 0; c < numClasses; c++) {
            // Relabellise a la volee pour CETTE classe seulement (une seule copie
            // de x vivante a la fois, au lieu de 10 copies simultanees) --
            // trainRawAll reste la seule source de x, jamais duplique par classe.
            vector<LabeledExample> trainBin;
            trainBin.reserve(trainRawAll.size());
            for (auto& e : trainRawAll) trainBin.push_back({ e.x, e.y == c ? 1 : 0 });

            trainBoostedClass(trainBin, n, K, N, p.S, p.a, total, discoverCount, noiseTol, M,
                              ensembles[c], alphas[c], purities[c]);
            cerr << "[RSS] apres classe " << c << " (" << (c+1) << "/" << numClasses << ") : "
                 << vmRssKB()/1024.0 << " Mo\n" << flush;
        }

        int correctFull = 0, correctBase = 0;
        for (auto& ex : testRawAll) {
            vector<double> scoreFull(numClasses, 0);
            for (int c = 0; c < numClasses; c++)
                for (size_t m = 0; m < ensembles[c].size(); m++)
                    scoreFull[c] += alphas[c][m] * normalizedScore(ensembles[c][m], ex.x);
            int predFull = 0;
            for (int c = 1; c < numClasses; c++) if (scoreFull[c] > scoreFull[predFull]) predFull = c;
            if (predFull == ex.y) correctFull++;

            vector<double> scoreBase(numClasses, 0);
            for (int c = 0; c < numClasses; c++)
                for (size_t m = 0; m < ensembles[c].size(); m++) {
                    int leaf = ensembles[c][m].router->route(ex.x);
                    scoreBase[c] += alphas[c][m] * purities[c][m][leaf];
                }
            int predBase = 0;
            for (int c = 1; c < numClasses; c++) if (scoreBase[c] > scoreBase[predBase]) predBase = c;
            if (predBase == ex.y) correctBase++;
        }
        accsFull[run]     = 100.0 * correctFull / testRawAll.size();
        accsBaseline[run] = 100.0 * correctBase / testRawAll.size();
        auto t1 = chrono::steady_clock::now();
        cout << "  run " << run << " : systeme_complet=" << accsFull[run]
             << "%  baseline_routeurs_ponderes=" << accsBaseline[run]
             << "%  [" << chrono::duration<double>(t1 - t0).count() << "s]\n" << flush;
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
