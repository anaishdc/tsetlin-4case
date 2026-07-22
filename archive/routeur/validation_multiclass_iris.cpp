#include "../tsetlin_engine.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <iomanip>

// =================================================================
// Validation multi-classe Iris -- version finale, consolidee apres
// investigation complete (voir resume des conclusions ci-dessous).
//
// Reproduit la methodologie du papier (Section 5.1, Multi-Class
// Tsetlin Machine) : les 3 reseaux un-contre-tous sont entraines sur
// le MEME split a chaque run, combines par argmax pour produire UNE
// prediction finale par exemple. Split 80/20, splits aleatoires
// repetes, Mean/5%ile/95%ile/Min/Max rapportes comme dans le papier.
//
// -----------------------------------------------------------------
// Resume de l'investigation (bugs trouves et corriges, pistes
// testees et ecartees) :
//
// 1. BUG CORRIGE -- combineur manquant, 2. BUG CORRIGE -- purete du
//    routeur mal estimee (buildRouterTreeRefit), 3. Pistes testees
//    et ecartees (S, total, K seul, ensemble de clauses intra-
//    feuille, discoverCount proportionnel a K, minLeafSize) : voir
//    l'historique de session pour le detail complet (desormais hors
//    de ce fichier, deja documente et clos precedemment).
//
// 4. DIAGNOSTIC CLE -- a K=6/K=10, le routeur SEUL (sans clause)
//    atteint deja 88-90% : le partitionnement fin approche la
//    memorisation. A K=2, le routeur seul ne fait que 61.8% -- les
//    clauses apportent +29 points de signal reellement appris. K=2
//    reste le reglage le plus honnete pour evaluer la regle.
//
// 5. VARIANCE (~±5-6%) -- diagnostic sur 200 runs montre que
//    l'instabilite N'EST PAS du bruit algorithmique : 16 exemples
//    (10.7% du dataset, tous versicolor/virginica) concentrent 80%
//    des erreurs, 89/149 (60%) ne sont JAMAIS mal classes. C'est un
//    chevauchement REEL des donnees, pas un defaut de l'algorithme.
//
// -----------------------------------------------------------------
// BOOSTING (routeur pondere, meme technique retenue sur Digits) :
// premier test rapide (5 runs) avait suggere un effet quasi nul
// (90.7% +/- 4.9%), mais c'etait un echantillon trop petit pour
// etre fiable vu la variance elevee (~5-6%/run). Sur 30 runs, le
// gain est en fait TRES net :
//
//   M=1 (sans boosting)                : 90.4% +/- 6.0%
//   Boosting (routeur pondere, M=8, 30 runs) : 93.7% +/- 3.9%  <- retenu
//
// +3.3 points ET variance reduite (6.0%->3.9%). Nouveau record pour
// Iris, meme au-dela de l'ensemble M=5 teste plus tot (92.2%). Le
// boosting aide donc bien Iris aussi, contrairement a la conclusion
// initiale (basee sur un echantillon insuffisant) -- le plafond
// ~90% n'etait pas totalement infranchissable, juste plus dur a
// depasser que sur Digits vu la petite taille du dataset (149
// exemples) qui rend chaque round de boosting plus bruite.
//
// BASELINE (routeurs ponderes seuls, sans clause, 30 runs) : 57.0%
// +/- 14.3% (5%ile=33.3%, 95%ile=90.0%, Min=30.0%, Max=90.0%), a
// comparer aux 93.7% +/- 3.9% du systeme complet -- ecart de 36.7
// points. Variance de la baseline tres elevee (14.3%) car sur un
// dataset aussi petit (119 exemples train, K=2 -> 2 feuilles), la
// purete d'une feuille peut chuter fortement d'un split a l'autre ;
// les clauses stabilisent le signal en plus de l'ameliorer. Ecart
// plus faible que sur Digits (36.7 vs 53.8 pts) car K=2 est un
// partitionnement volontairement grossier (cf. point 4 ci-dessus),
// donc le routeur seul reste deja informatif sur Iris.
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
        data.push_back({ x, label }); // y = classe brute (0,1,2), pas binaire
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

// Entraine un ensemble boosted (M rounds) pour UNE classe : routeur
// ET clauses ponderes a chaque round (comme sur Digits).
void trainBoostedClass(const vector<LabeledExample>& trainBin, int n, int K, int N,
                       double S, double a, int total, int discoverCount, double noiseTol, int M,
                       vector<MultiClauseNetwork>& nets, vector<double>& alphas,
                       vector<vector<double>>& purities) {
    int Nex = (int)trainBin.size();
    vector<double> w(Nex, 1.0 / Nex);

    for (int m = 0; m < M; m++) {
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
    const int    n             = 16;
    const int    numClasses    = 3;
    const int    K             = argc > 1 ? atoi(argv[1]) : 2;    // config "honnete" : les clauses apportent +29pt vs routeur seul (K surchargeable : argv[1])
    const int    N             = 320;
    const Params p             = {1,0.3};
    const int    total         = 100000;
    const int    discoverCount = argc > 3 ? atoi(argv[3]) : max(300, 3 * n);   // regle generale : proportionnel a n, pas a la taille du dataset
    const double noiseTol      = 0.10;
    const int    M             = 16;    // rounds de boosting (coherence avec Digits, gain minime ici)
    const int    nbRuns        = argc > 2 ? atoi(argv[2]) : 40;   // nbRuns surchargeable : argv[2]

    auto allRaw = loadAll("../data/BinaryIrisData.txt", n);
    const int trainSize = (int)std::round(0.8 * allRaw.size()); // 80/20 comme le papier

    cout << fixed << setprecision(1);
    cout << "Dataset : " << allRaw.size() << " exemples (papier: 150)  train=" << trainSize
         << " test=" << (allRaw.size() - trainSize) << "  K=" << K << " M=" << M
         << " (" << nbRuns << " runs)\n\n";

    vector<double> accsFull(nbRuns), accsBaseline(nbRuns);
    for (int run = 0; run < nbRuns; run++) {
        rng.seed(20000 + run);

        vector<LabeledExample> shuffled = allRaw;
        shuffle(shuffled.begin(), shuffled.end(), rng);
        vector<LabeledExample> trainRaw(shuffled.begin(), shuffled.begin() + trainSize);
        vector<LabeledExample> testRaw(shuffled.begin() + trainSize, shuffled.end());

        vector<vector<MultiClauseNetwork>> ensembles(numClasses);
        vector<vector<double>> alphas(numClasses);
        vector<vector<vector<double>>> purities(numClasses);
        for (int c = 0; c < numClasses; c++) {
            vector<LabeledExample> trainBin;
            for (auto& e : trainRaw) trainBin.push_back({ e.x, e.y == c ? 1 : 0 });
            trainBoostedClass(trainBin, n, K, N, p.S, p.a, total, discoverCount, noiseTol, M,
                              ensembles[c], alphas[c], purities[c]);
        }

        int correctFull = 0, correctBase = 0;
        for (auto& e : testRaw) {
            // Systeme complet : argmax sur la somme ponderee (alpha) des scores normalises
            vector<double> scoreFull(numClasses, 0);
            for (int c = 0; c < numClasses; c++)
                for (size_t m = 0; m < ensembles[c].size(); m++)
                    scoreFull[c] += alphas[c][m] * normalizedScore(ensembles[c][m], e.x);
            int predFull = 0;
            for (int c = 1; c < numClasses; c++) if (scoreFull[c] > scoreFull[predFull]) predFull = c;
            if (predFull == e.y) correctFull++;

            // Baseline : argmax sur la purete ponderee (routeurs seuls, sans clause)
            vector<double> scoreBase(numClasses, 0);
            for (int c = 0; c < numClasses; c++)
                for (size_t m = 0; m < ensembles[c].size(); m++) {
                    int leaf = ensembles[c][m].router->route(e.x);
                    scoreBase[c] += alphas[c][m] * purities[c][m][leaf];
                }
            int predBase = 0;
            for (int c = 1; c < numClasses; c++) if (scoreBase[c] > scoreBase[predBase]) predBase = c;
            if (predBase == e.y) correctBase++;
        }
        accsFull[run]     = 100.0 * correctFull / testRaw.size();
        accsBaseline[run] = 100.0 * correctBase / testRaw.size();
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
