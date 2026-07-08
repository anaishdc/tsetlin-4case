#include "tsetlin_engine.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>

// =================================================================
// Noisy XOR : boosting AdaBoost avec routeur
// PONDERE reconstruit a chaque round ,
// combineur par somme ponderee des scores normalises, et
// baseline "routeurs ponderes seuls" (purete, sans clause) pour
// verifier que les clauses apportent bien quelque chose.
//
// 12 features (x1,x2 = signal XOR, x3..x12 = bruit pur), label
// binaire. Train ~40% de bruit sur le label, test 0% 
//
// RESULTAT (10 runs) : systeme complet = 100.0% +/- 0.0% (10/10 runs
// parfaits) vs baseline (routeurs ponderes seuls) = 50.1% +/- 0.0%
// (= niveau du hasard). Ecart de 49.9 points 
// =================================================================


// =================================================================
// chargement des données 
// =================================================================
vector<LabeledExample> loadDataset(const string& path, int nFeatures) {
    vector<LabeledExample> data;
    ifstream f(path);
    string line;
    while (getline(f, line)) {
        if (line.empty()) continue;
        istringstream iss(line);
        vector<int> x(nFeatures);
        for (int i = 0; i < nFeatures; i++) iss >> x[i];
        int y; iss >> y;
        data.push_back({ x, y });
    }
    return data;
}


// =================================================================
// le calcul du score normalisé 
// =================================================================
double normalizedScore(const MultiClauseNetwork& net, const vector<int>& x) {
    int leaf = net.router->route(x); // trouver la bonne feuille pour x 
    const Clause& c = net.clauses[leaf]; // on recupere la clause associé à cette feuille 
    int satisfied = 0, violated = 0, nLit = 0; // calculer le nombre de litteraux satifait violé et total 
    for (int i = 0; i < c.n; i++) {
        if (c.v[i].included())    { nLit++; x[i] == 1 ? satisfied++ : violated++; }
        if (c.vbar[i].included()) { nLit++; x[i] == 0 ? satisfied++ : violated++; }
    }
    int rawScore = violated > 0 ? -violated : satisfied;
    return nLit > 0 ? (double)rawScore / nLit : 0.0; // on normalise 
}


// =================================================================
// la baseline (l'arbre tout seul)
// Elle calcule la proportion de "1" dans chaque feuille en tenant compte des poids.
// =================================================================

vector<double> leafPuritiesWeighted(const RouterTree& tree, const vector<LabeledExample>& trainBin,
                                    const vector<double>& w) {
    int K = (int)tree.leafUsedFeatures.size(); // nombre de feuille 
    vector<double> wSum(K, 0.0), wPos(K, 0.0); // la masse totale et la masse des exemples a 1 
    for (size_t i = 0; i < trainBin.size(); i++) {
        int leaf = tree.route(trainBin[i].x);
        wSum[leaf] += w[i];
        if (trainBin[i].y == 1) wPos[leaf] += w[i];
    }
    vector<double> purity(K, 0.0);
    for (int k = 0; k < K; k++) purity[k] = wSum[k] > 0 ? wPos[k] / wSum[k] : 0.0;
    return purity;
}


// =================================================================
// Sélectionne un exemple : équilibré à 50/50 pour la classe (y),
// et favorise les exemples à fort poids (les plus difficiles)
// =================================================================
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



// =================================================================
// Entraine un ensemble boosted (M rounds) : routeur ET clauses ponderes
// =================================================================


void trainBoostedClass(const vector<LabeledExample>& trainBin, int n, int K, int N,
                       double S, double a, int total, int discoverCount, double noiseTol, int M,
                       vector<MultiClauseNetwork>& nets, vector<double>& alphas,
                       vector<vector<double>>& purities) {

    int Nex = (int)trainBin.size(); // Récupère le nombre total d'exemples dans le dataset 
    vector<double> w(Nex, 1.0 / Nex); // Crée le tableau des poids : tout le monde commence à égalité (1 / Nex) 

    //  LA BOUCLE DU BOOSTING (M ROUNDS)
    for (int m = 0; m < M; m++) {
        // Création d'un tableau d'indices allant de 0 à Nex-1 
        vector<int> idx(Nex);
        for (int i = 0; i < Nex; i++) idx[i] = i;

        // TRI DES EXEMPLES : On trie les indices pour mettre les exemples les plus LOURDS au début
        sort(idx.begin(), idx.end(), [&](int a2, int b2) { return w[a2] > w[b2]; });
        
        int dc = min(Nex, discoverCount); // on prend soit la taille du dataset, soit discoverCount (le max d'exemples pour l'arbre)
       

        // Création du sous-échantillon (batch) contenant uniquement les exemples les plus difficiles
         vector<WeightedLabeledExample> discoverBatch;
        for (int i = 0; i < dc; i++) discoverBatch.push_back({ trainBin[idx[i]].x, trainBin[idx[i]].y, w[idx[i]] });

        //FABRICATION DE L'ARBRE 
        //  On donne notre lot de données difficiles à l'arbre pour qu'il calcule ses axes de coupure
        auto router = buildWeightedRouterTree(discoverBatch, n, K, noiseTol);
        // On calcule et stocke la pureté pondérée des feuilles de cet arbre (utile pour la baseline)
        purities.push_back(leafPuritiesWeighted(*router, trainBin, w));
        // On crée un nouveau réseau de Tsetlin couplé avec cet arbre et on l'ajoute à la liste des modèles
        nets.emplace_back(n, N, K, std::move(router), S, a);

        
        WeightedSampler sampler(trainBin, w);
        // on pioche 100 000 fois un exemple difficile et on l'envoie aux automates
        for (int t = 0; t < total; t++) {
            const auto& ex = sampler.sample();
            nets.back().update(ex.x, ex.y);
        }

        // CALCUL DE L'ERREUR GLOBALE PONDÉRÉE 
        double err = 0, wSum = 0;
        vector<bool> wrong(Nex); // Tableau de booléens pour enregistrer qui a eu faux
        
        // On teste le modèle tout neuf sur l'INTÉGRALITÉ du dataset d'entraînement
        for (int i = 0; i < Nex; i++) {
            // Est-ce que le réseau se trompe sur la prédiction de l'exemple i ?
            wrong[i] = (nets.back().output(trainBin[i].x) != trainBin[i].y);
            
            if (wrong[i]) err += w[i]; // Si le modèle a FAUX, on accumule le POIDS de l'exemple dans l'erreur
            wSum += w[i];              // Accumulation de la somme totale des poids (vaut normalement 1.0)
        }
        
        // Sécurité mathématique pour éviter d'avoir une erreur pile égale à 0 ou à 1 (ce qui casserait le log)
        err = max(1e-6, min(0.999, err / wSum));
        
       
        // Formule officielle AdaBoost : plus l'erreur est faible, plus alpha est grand
        double alpha = 0.5 * log((1.0 - err) / err);
        alphas.push_back(alpha); // On enregistre l'alpha de ce round

        //  MISE À JOUR DES POIDS POUR LE PROCHAIN ROUND 
        double norm = 0;
        for (int i = 0; i < Nex; i++) { 
            // Si faux -> multiplié par exp(alpha) [Poids AUGMENTE]
            // Si juste -> multiplié par exp(-alpha) [Poids DIMINUE]
            w[i] *= wrong[i] ? exp(alpha) : exp(-alpha); 
            norm += w[i]; // On calcule la nouvelle somme totale des poids pour pouvoir normaliser
        }
        
        // Normalisation : on divise chaque poids par la somme pour que le total de w fasse à nouveau 1.0
        for (int i = 0; i < Nex; i++) w[i] /= norm;
    }
}




int main(int argc, char** argv) {
    // ---- HYPERPARAMETRES ----
    const int    n             = 12; // nombre de features
    const int    K             = argc > 1 ? atoi(argv[1]) : 2;  // nombre de clauses (surchargeable : argv[1])
    const int    N             = 500 ; // nombre d'etat d'automate
    const Params p             = {1,1,1,1,0.3,0.3}; //parametres par defaut
    const int    total         = 30000; // nombre de tirages d'entrainement par round de boosting
    const int    discoverCount = argc > 3 ? atoi(argv[3]) : max(300, 3 * n);   // le nombre d'exemple utilisé pour construire le routeur  proportionnel a n, pas a la taille du dataset
    const double noiseTol      = 0.40; // ~40% de bruit connu sur le label (train)
    const int    M             = 5;    // rounds de boosting
    const int    nbRuns        = argc > 2 ? atoi(argv[2]) : 20;    // nombre de repetitions independantes (surchargeable : argv[2])

    // Chargement des jeux d'entrainement et de test depuis les fichiers du XOR bruite
    auto trainData = loadDataset("data/NoisyXORTrainingData.txt", n);
    auto testData  = loadDataset("data/NoisyXORTestData.txt",     n);

    cout << fixed << setprecision(1);
    cout << "Dataset : train=" << trainData.size() << " test=" << testData.size()
         << "  K=" << K << " M=" << M << " (" << nbRuns << " runs)\n\n";

    // Accuracies (une par run) du systeme complet et de la baseline, pour calculer moyenne/ecart-type
    vector<double> accsFull(nbRuns), accsBaseline(nbRuns);
    for (int run = 0; run < nbRuns; run++) {
        rng.seed(30000 + run); // seed differente a chaque run pour avoir des resultats independants

        // Entrainement complet du boosting : produit M reseaux (arbre+clauses), leurs poids alpha,
        // et les puretes de feuilles (necessaires seulement pour la baseline)
        vector<MultiClauseNetwork> nets;
        vector<double> alphas;
        vector<vector<double>> purities;
        trainBoostedClass(trainData, n, K, N, p.S1, p.a1, total, discoverCount, noiseTol, M,
                          nets, alphas, purities);

        // ---- EVALUATION SUR LE TEST SET ----
        int correctFull = 0, correctBase = 0;
        for (auto& ex : testData) {
            double scoreFull = 0, scoreBase = 0;
            // On cumule les votes ponderes (par alpha) de chaque round de boosting
            for (size_t m = 0; m < nets.size(); m++) {
                // Systeme complet : score base sur la clause associee a la feuille
                scoreFull += alphas[m] * normalizedScore(nets[m], ex.x);
                // Baseline : score base uniquement sur la purete de la feuille (sans clause)
                int leaf = nets[m].router->route(ex.x);
                scoreBase += alphas[m] * (purities[m][leaf] - 0.5); // recentre autour de 0 comme un score
            }
            // Vote final : signe du score cumule
            int predFull = scoreFull > 0 ? 1 : 0;
            int predBase = scoreBase > 0 ? 1 : 0;
            if (predFull == ex.y) correctFull++;
            if (predBase == ex.y) correctBase++;
        }
        accsFull[run]     = 100.0 * correctFull / testData.size();
        accsBaseline[run] = 100.0 * correctBase / testData.size();
        cout << "  run " << run << " : systeme_complet=" << accsFull[run]
             << "%  baseline_routeurs_ponderes=" << accsBaseline[run] << "%\n" << flush;
    }

    // Calcule et affiche moyenne, ecart-type, percentiles 5/95 et min/max sur les nbRuns accuracies
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
    // Ecart de performance = apport reel des clauses par rapport aux routeurs ponderes seuls
    cout << "Ecart (apport reel des clauses)  : " << (meanFull - meanBase) << " points\n";

    return 0;
}
