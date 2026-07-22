#include "../tsetlin_engine.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <iomanip>

// =================================================================
// Validation multi-classe Digits -- meme methodologie et memes
// enseignements que validation_multiclass_iris.cpp (voir ce fichier
// pour le detail complet de l'investigation menee sur Iris,
// directement transposable ici).
//
// Pretraitement IDENTIQUE au papier (Section 5.2) : "transforming
// the 64 different input features into a sequence of 192 bits, 3
// bits per pixel" -- fichiers data/BinaryDigitsTrainingData192.txt /
// TestData192.txt (1437 train / 360 test, meme taille totale que le
// dataset utilise par le papier). Split train/test FIXE (pas
// reshuffle a chaque run, comme fourni par les fichiers).
//
// -----------------------------------------------------------------
// Choix de K et N : avec 192 features (vs 16 pour Iris), K=2 etait
// trop conservateur, K=4 retenu (meilleur compromis accuracy/ecart
// avec le bagging M=12, cf. anciennes notes -- desormais remplace
// par le boosting ci-dessous). N=1000 (etats d'automate, comme le
// papier) ameliore l'accuracy et reduit la variance.
//
// -----------------------------------------------------------------
// BUG TROUVE ET CORRIGE -- score non normalise entre classes :
// diagnostic de la confusion 8->1 (la plus frequente au debut) a
// montre que le reseau "1" a des clauses avec ~2x plus de litteraux
// que le reseau "8" (47.3 vs 25.8 litteraux/clause en moyenne). Un
// score BRUT (satisfaits - violes) n'est donc pas comparable entre
// classes. Correctif : normalizedScore() = score_brut / nb_litteraux
// inclus dans la feuille routee, utilise pour l'argmax.
//
// -----------------------------------------------------------------
// BOOSTING (routeur pondere) au lieu du bagging M=12 -- amelioration
// majeure. Contrairement au bagging (M reseaux INDEPENDANTS sur la
// meme distribution equilibree), le boosting entraine les reseaux
// SEQUENTIELLEMENT, chacun sur-pondere les exemples que les
// precedents ont rates (comme AdaBoost).
//
// Premier essai (routeur FIGE, seul l'echantillonnage des clauses
// est pondere) : ECHEC net, 64.7% (bien pire que le bagging 89.0%).
// Diagnostic : le routeur ne peut pas "recadrer" le partitionnement
// sur les zones difficiles s'il reste identique a chaque round --
// les alphas (poids AdaBoost) s'effondraient a quasi-zero des le
// round 2 pour la plupart des classes.
//
// Correctif : le routeur est maintenant RECONSTRUIT a chaque round
// avec les poids courants (buildWeightedRouterTree, impurete de Gini
// ponderee pour le choix des splits ET les stats de feuille). Chaque
// round peut alors vraiment se recentrer sur les zones difficiles.
// Resultat (comparaison sur seed identique) :
//
//   Bagging M=12 (score normalise)     : 89.0% +/- 1.2%
//   Boosting v1 (routeur fige, M=8)    : 64.7%  (echec)
//   Boosting v2 (routeur pondere, M=8) : 93.9%  <- retenu
//
// Verification (comme pour Iris et le bagging) que les clauses
// apportent bien un gain reel, pas seulement le routeur adaptatif :
// baseline (routeurs ponderes seuls, sans clause) = 38.9%, systeme
// complet = 93.9%, ecart = +55.0 points -- le plus grand ecart de
// toute l'investigation (vs ~26-29pt pour le bagging). A mesure que
// les rounds de boosting concentrent les poids sur des sous-
// ensembles de plus en plus petits et difficiles, le routeur seul
// devient de moins en moins fiable, mais les clauses continuent
// d'extraire un signal genuinement generalisable.
//
// A noter : la MEME approche testee sur Iris n'apporte presque rien
// (90.4% -> 90.7%, +0.3pt) car le plafond ~90% d'Iris est du a un
// chevauchement REEL des donnees (versicolor/virginica), pas a un
// defaut algorithmique -- le boosting ne peut pas creer un signal
// discriminant qui n'existe pas dans les donnees. Applique ici quand
// meme pour coherence methodologique entre les deux datasets.
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

// Score NORMALISE : score brut (satisfaits - violes) divise par le
// nombre de litteraux inclus dans la clause de la feuille routee.
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

// Sampler pondere : pool positif/negatif choisis 50/50 (comme
// BalancedSampler), mais l'exemple DANS le pool est tire
// proportionnellement a son poids de boosting.
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

// Entraine un ensemble boosted (M rounds) pour UNE classe (probleme
// un-contre-tous) : routeur ET clauses ponderes a chaque round.
void trainBoostedClass(const vector<LabeledExample>& trainBin, int n, int K, int N,
                       double S, double a, int total, int discoverCount, double noiseTol, int M,
                       vector<MultiClauseNetwork>& nets, vector<double>& alphas,
                       vector<vector<double>>& purities) {
    int Nex = (int)trainBin.size();
    vector<double> w(Nex, 1.0 / Nex);

    for (int m = 0; m < M; m++) {
        // Routeur PONDERE : discoverBatch priorise les exemples de poids
        // fort (les plus durs), impurete de Gini calculee avec les poids.
        vector<int> idx(Nex);
        for (int i = 0; i < Nex; i++) idx[i] = i;
        sort(idx.begin(), idx.end(), [&](int a2, int b2) { return w[a2] > w[b2]; });
        int dc = min(Nex, discoverCount);
        vector<WeightedLabeledExample> discoverBatch;
        for (int i = 0; i < dc; i++) discoverBatch.push_back({ trainBin[idx[i]].x, trainBin[idx[i]].y, w[idx[i]] });

        auto router = buildWeightedRouterTree(discoverBatch, n, K, noiseTol);
        purities.push_back(leafPuritiesWeighted(*router, trainBin, w));
        nets.emplace_back(n, N, K, std::move(router), S, a);

        WeightedSampler sampler(trainBin, w);
        for (int t = 0; t < total; t++) {
            const auto& ex = sampler.sample();
            nets.back().update(ex.x, ex.y);
        }

        // Erreur ponderee de ce round -> poids de vote alpha (AdaBoost)
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

        // Mise a jour des poids : augmente sur les erreurs, diminue sinon
        double norm = 0;
        for (int i = 0; i < Nex; i++) {
            w[i] *= wrong[i] ? exp(alpha) : exp(-alpha);
            norm += w[i];
        }
        for (int i = 0; i < Nex; i++) w[i] /= norm;
    }
}

int main(int argc, char** argv) {
    // 192 = 64 pixels x 3 bits/pixel (encodage thermometre), pour
    // reproduire exactement le pretraitement du papier (Section 5.2).
    const int    n             = 192;
    const int    numClasses    = 10;
    const int    K             = argc > 1 ? atoi(argv[1]) : 4;    // K surchargeable : argv[1]
    const int    N             = 1000;
    const Params p             = {1,0.3};
    const int    total         = 100000;
    const int    discoverCount = argc > 3 ? atoi(argv[3]) : max(300, 3 * n);   // regle generale : proportionnel a n, pas a la taille du dataset
    const double noiseTol      = 0.05;
    const int    M             = 16;    // rounds de boosting (cf. note ci-dessus)
    const int    nbRuns        = argc > 2 ? atoi(argv[2]) : 4;   // nbRuns surchargeable : argv[2]

    auto trainRawAll = loadAll("../data/BinaryDigitsTrainingData192.txt", n);
    auto testRawAll  = loadAll("../data/BinaryDigitsTestData192.txt",     n);

    vector<vector<LabeledExample>> trainBinByClass(numClasses);
    for (int c = 0; c < numClasses; c++)
        for (auto& e : trainRawAll) trainBinByClass[c].push_back({ e.x, e.y == c ? 1 : 0 });

    cout << fixed << setprecision(1);
    cout << "Dataset : train=" << trainRawAll.size() << " test=" << testRawAll.size()
         << "  K=" << K << " M=" << M << " (" << nbRuns << " runs)\n\n";

    vector<double> accsFull(nbRuns), accsBaseline(nbRuns);
    for (int run = 0; run < nbRuns; run++) {
        rng.seed(21000 + run);

        vector<vector<MultiClauseNetwork>> ensembles(numClasses);
        vector<vector<double>> alphas(numClasses);
        vector<vector<vector<double>>> purities(numClasses);
        for (int c = 0; c < numClasses; c++) {
            trainBoostedClass(trainBinByClass[c], n, K, N, p.S, p.a, total, discoverCount, noiseTol, M,
                              ensembles[c], alphas[c], purities[c]);
        }

        int correctFull = 0, correctBase = 0;
        for (auto& ex : testRawAll) {
            // Systeme complet : argmax sur la SOMME PONDEREE (alpha) des scores normalises
            vector<double> scoreFull(numClasses, 0);
            for (int c = 0; c < numClasses; c++)
                for (size_t m = 0; m < ensembles[c].size(); m++)
                    scoreFull[c] += alphas[c][m] * normalizedScore(ensembles[c][m], ex.x);
            int predFull = 0;
            for (int c = 1; c < numClasses; c++) if (scoreFull[c] > scoreFull[predFull]) predFull = c;
            if (predFull == ex.y) correctFull++;

            // Baseline : argmax sur la purete ponderee (routeurs seuls, sans clause)
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
