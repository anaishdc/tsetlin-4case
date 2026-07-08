#pragma once

#include "1_clause/tsetlin_engine_4case.h"
#include <memory>



// Exemple labellise : ([1,0,1,0,0], 0)
struct LabeledExample {
    vector<int> x;
    int y;
};

// ---------------------------------------------------------------
// Routeur : arbre de decision fige une fois construit,
// entrée  : x -> sortie :  indice de clause.
// ---------------------------------------------------------------


// =================================================================
//  Structure d'un noeud de l'arbre : 
// =================================================================
struct RouterNode {
    bool isLeaf       = false;           // si c'est une feuille 
    int  splitFeature = -1;              // si noeud interne : feature de separation
    int  clauseIndex  = -1;              // si feuille : indice de la clause associee
    unique_ptr<RouterNode> left, right;  // gauche si x[feature]==0, droite si x[feature]==1
};


// =================================================================
// Structure de routeur complet 
// =================================================================

struct RouterTree {
    unique_ptr<RouterNode> root; // la racine 

    // leafUsedFeatures[k][i] = vrai si la feature i a servi a router
    // vers la feuille k 
    vector<vector<bool>> leafUsedFeatures;

    // leafFixedValue[k][i] = valeur imposee par le chemin pour i
    // (valide seulement si leafUsedFeatures[k][i]=vrai)
    vector<vector<int>> leafFixedValue;

    // leafIsPureZero[k] = vrai si fraction d'exemples y=1 <= noiseTol
    vector<bool> leafIsPureZero;

    // leafMajorityVote[k] = vote majoritaire de la feuille (baseline
    // "routeur seul", sans aucune clause)
    vector<int> leafMajorityVote;

    // leafWorstSkew[k] = min(q,1-q) sur la feature non figee la plus
    // desequilibree de la feuille k
    vector<double> leafWorstSkew;

    // fonction de routage d'un exemple 
    int route(const vector<int>& x) const {
        RouterNode* node = root.get();
        while (!node->isLeaf)
            node = (x[node->splitFeature] == 0) ? node->left.get() : node->right.get();
        return node->clauseIndex;
    }

    // vote majoritaire 
    int routeMajority(const vector<int>& x) const { return leafMajorityVote[route(x)]; }
};


// =================================================================
// certaines features peuvent etre toujours a 0 (comme dans MNIST) donc
// notre regle finit par les inclure meme si elles ne devraient jamais
// l'etre , cette fonction detecte le risque et le corrige.
// =================================================================

inline pair<double, double> calibrateParams(double baseS, double baseA, double worstSkew) {
    const double margin = 4.0;
    // 1 er cas : la feature est bien equilibrée : aucun ajustement n'est necessaire  
    if (worstSkew >= 0.40) return { baseS, baseA };

    double B     = (1.0 - worstSkew) / (1.0 - 2.0 * worstSkew); // condition de convergence vers exclude 
    double ratio = B / margin;
    if (ratio >= baseS / baseA) return { baseS, baseA };
    double a = 1.0;
    double S = ratio * a;
    return { S, a };
}

// =================================================================
// Reprend la table de la regle a 4 cas, en sautant les features
// figees par le chemin du routeur (deja decidees, pas d'apprentissage).
// =================================================================

inline void updateMasked(Clause& clause, const vector<int>& x, int y,
                         const Params& p, const vector<bool>& active) {
    double S = p.S1, a = p.a1;
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

// ---------------------------------------------------------------
// Reseau multi-clause : K clauses + un routeur fige.
// ---------------------------------------------------------------
struct MultiClauseNetwork {

    vector<Clause>       clauses;  // enseble de clauses 
    vector<vector<bool>> active;   // active[k][i] = clause k apprend sur la feature i
    vector<Params>       params;   // params[k] : (S,a) calibres pour la feuille k
    unique_ptr<RouterTree> router; // le routeur 

    MultiClauseNetwork(int n, int N, int K, unique_ptr<RouterTree> r, double baseS, double baseA)
        : router(std::move(r)) {
        int Kactual = (int)router->leafUsedFeatures.size(); // <= K : le routeur s'arrete tot si resolu

        // on figue les features utilisés dans le routage pour chaque clause ( donc feuille )
        for (int k = 0; k < Kactual; k++) {
            clauses.emplace_back(n, N);
            vector<bool> act(n, true); // vecteur pour savoir si on doit apprendre l'automate de la feature 
            bool forcedLiteral = false;

            for (int i = 0; i < n; i++) {

                // si la feature n'est pas utilisé dans le routage on continue 
                if (!router->leafUsedFeatures[k][i]) continue;

                act[i] = false;

                // Si la feuille est PureZero, on force UN littoral (le
                // premier de la feuille) a bloquer definitivement le
                // motif y=1 -- le reste des features figees est exclu.

                if (router->leafIsPureZero[k] && !forcedLiteral) {
                
                    if (router->leafFixedValue[k][i] == 1) {
                        clauses[k].vbar[i].state = 2 * N; // inclu
                        clauses[k].v[i].state    = 1; //exlu 
                    } else {
                        clauses[k].v[i].state    = 2 * N; // inclu 
                        clauses[k].vbar[i].state = 1; // exclu 
                    }
                    forcedLiteral = true;
                // dans les cas standard on fixe leur automates à exlude    
                } else {
                    clauses[k].v[i].state    = 1;
                    clauses[k].vbar[i].state = 1;
                }
            }

            active.push_back(act);
            // pour chaque clause on trouve les bons params alpha et s 
            auto [S, a] = calibrateParams(baseS, baseA, router->leafWorstSkew[k]);
            params.push_back({ S, S, S, S, a, a });
        }
    }


    // pour un exemple  donné on le route vers la bonne clause puis on calclure la valeur de sortie a partir 
    // des etats des automates de cette clause 
    int output(const vector<int>& x) const {
        return clauses[router->route(x)].output(x);
    }

    // Score de confiance : clause vide , violee , satisfaite.
    int score(const vector<int>& x) const {
        const Clause& c = clauses[router->route(x)]; // router l'exemple vers labonne clause
        int satisfied = 0, violated = 0; // nombre de litteraux satisfaits et violés
        for (int i = 0; i < c.n; i++) {
            if (c.v[i].included())    { x[i] == 1 ? satisfied++ : violated++; }
            if (c.vbar[i].included()) { x[i] == 0 ? satisfied++ : violated++; }
        }
        return violated > 0 ? -violated : satisfied;
    }

    // mise à jour des automates d'une clause sur es features non utilisé dans le routage 
    void update(const vector<int>& x, int y) {
        int idx = router->route(x);
        updateMasked(clauses[idx], x, y, params[idx], active[idx]);
    }
};



// ---------------------------------------------------------------
// Routeur PONDERE : chaque exemple porte un poids (boosting).
// ---------------------------------------------------------------


// =================================================================
// Si le modèle précédent s'est trompé sur cet exemple, le boosting 
// va augmenter w. Plus w est grand, plus l'arbre va faire d'efforts 
// pour ne pas se tromper sur cet exemple précis lors de la création 
// des prochaines branches.
// =================================================================

struct WeightedLabeledExample {
    vector<int> x; // features 
    int y; // cible 
    double w; // le poids de l'exemple (introduit par le boosting)
};


// Le calcul de l'impureté de Gini pondérée
inline double weightedGiniImpurity(const vector<const WeightedLabeledExample*>& ex) {
    if (ex.empty()) return 0.0;
    double wSum = 0, wPos = 0;
    // on addition les poids de tous les exemples  , aisi que la somme des exemples avec y=1 
    for (auto* e : ex) { wSum += e->w; if (e->y == 1) wPos += e->w; }
    if (wSum <= 0) return 0.0;
    double p1 = wPos / wSum; // proportion ponderée d'exemples de la classe 1 
    return 2.0 * p1 * (1.0 - p1);
}


inline double computeGini(double wSum, double wPos) {
    if (wSum <= 0) return 0.0;
    double p1 = wPos / wSum;
    return 2.0 * p1 * (1.0 - p1);
}
 
// Structure d'une feuille en cours de construction 
struct WLeafInfo {
    RouterNode* node;        // Un pointeur vers le nœud correspondant dans l'arbre
    vector<const WeightedLabeledExample*> examples; // La liste des exemples qui ont atterri dans cette feuille
    vector<bool> usedFeatures; // Quelles features ont déjà été utilisées pour arriver ici
    vector<int>  fixedValue; // Les valeurs imposées par le chemin
};


struct LookaheadBuffers {
    vector<double> lSum0, lPos0, lSum1, lPos1; // Buffers pour le côté GAUCHE (l = left) de la première séparation f
    vector<double> rSum0, rPos0, rSum1, rPos1; // Buffers pour le côté DROIT (r = right) de la première séparation f
    LookaheadBuffers(int n) : lSum0(n), lPos0(n), lSum1(n), lPos1(n),
                              rSum0(n), rPos0(n), rSum1(n), rPos1(n) {}
};


// calcule le score de pureté final pour une feature f donnée en simulant deux coups d'avance (Lookahead)
inline double lookaheadImpurityForFeature(const vector<const WeightedLabeledExample*>& ex,
                                          const vector<bool>& used, int n, int f,
                                          LookaheadBuffers& buf, double& outWL, double& outWR) {
          
    // on efface les chiffres de la partie précédente
    fill(buf.lSum0.begin(), buf.lSum0.end(), 0.0); fill(buf.lPos0.begin(), buf.lPos0.end(), 0.0);
    fill(buf.lSum1.begin(), buf.lSum1.end(), 0.0); fill(buf.lPos1.begin(), buf.lPos1.end(), 0.0);
    fill(buf.rSum0.begin(), buf.rSum0.end(), 0.0); fill(buf.rPos0.begin(), buf.rPos0.end(), 0.0);
    fill(buf.rSum1.begin(), buf.rSum1.end(), 0.0); fill(buf.rPos1.begin(), buf.rPos1.end(), 0.0);
    double wL = 0, wPosL = 0, wR = 0, wPosR = 0;

    // remplissage du buffer en parcourant chaque exemple 
    for (auto* e : ex) {
        bool isPos = (e->y == 1);
        double w = e->w;
        // coté gauche 
        if (e->x[f] == 0) {
            wL += w; if (isPos) wPosL += w;
            for (int g = 0; g < n; g++) {
                if (g == f || used[g]) continue;
                if (e->x[g] == 0) { buf.lSum0[g] += w; if (isPos) buf.lPos0[g] += w; }
                else              { buf.lSum1[g] += w; if (isPos) buf.lPos1[g] += w; }
            } // coté droit 
        } else {
            wR += w; if (isPos) wPosR += w;
            for (int g = 0; g < n; g++) {
                if (g == f || used[g]) continue;
                if (e->x[g] == 0) { buf.rSum0[g] += w; if (isPos) buf.rPos0[g] += w; }
                else              { buf.rSum1[g] += w; if (isPos) buf.rPos1[g] += w; }
            }
        }
    }

    // recherche de la meilleur 2 eme coupure 
    outWL = wL; outWR = wR;
    if (wL <= 0 || wR <= 0) return -1.0;

    // calcule l'impurté de base 
    double leftBest  = computeGini(wL, wPosL);
    double rightBest = computeGini(wR, wPosR);

    // teste virtuellement toutes les features  possibles
    for (int g = 0; g < n; g++) {
        if (g == f || used[g]) continue; // si déja utilisé on continue 
        if (buf.lSum0[g] > 0 && buf.lSum1[g] > 0) {
            double imp = (buf.lSum0[g] / wL) * computeGini(buf.lSum0[g], buf.lPos0[g])
                       + (buf.lSum1[g] / wL) * computeGini(buf.lSum1[g], buf.lPos1[g]);
            leftBest = min(leftBest, imp);
        }
        if (buf.rSum0[g] > 0 && buf.rSum1[g] > 0) {
            double imp = (buf.rSum0[g] / wR) * computeGini(buf.rSum0[g], buf.rPos0[g])
                       + (buf.rSum1[g] / wR) * computeGini(buf.rSum1[g], buf.rPos1[g]);
            rightBest = min(rightBest, imp);
        }
    }
    return (wL / (wL + wR)) * leftBest + (wR / (wL + wR)) * rightBest;
}

// Construit le routeur en ponderant chaque exemple (impurete de Gini
// ponderee pour le choix des splits ET pour les stats de feuille).

inline unique_ptr<RouterTree> buildWeightedRouterTree(const vector<WeightedLabeledExample>& data,
                                                      int n, int K, double noiseTol) {

    // creation d'un routeur vide (la racine est une feuille )
    auto tree = make_unique<RouterTree>();
    tree->root = make_unique<RouterNode>();
    tree->root->isLeaf = true;
    tree->root->clauseIndex = 0;

    // creation d'un tableau de pointeur vers tous les exemples du dataset 
    vector<const WeightedLabeledExample*> all;
    for (auto& e : data) all.push_back(&e);

   
    vector<WLeafInfo> leaves; // creation d'un tableau de feuille 
    // placer tous les exemples dans la racine
    leaves.push_back({ tree->root.get(), all, vector<bool>(n, false), vector<int>(n, 0) });

    const int LOOKAHEAD_TOPM = 30;

    // condition d'arret :  quand le nombre de feuille = k 
    // a chaque iteration on va  ajouter une feuille en coupant en deux 
    while ((int)leaves.size() < K) {

         
        int    bestLeafIdx = -1, bestFeature = -1;
        double bestGain    = -1.0;

        // on parcours les feuilles une par une 
        for (size_t li = 0; li < leaves.size(); li++) {
            auto& leaf = leaves[li];
            double parentImpurity = weightedGiniImpurity(leaf.examples); // calcule l'impureté de Gini de la feuille avant toute coupure
            double leafMass = 0; // la somme des poids de la feuille 
            
            for (auto* e : leaf.examples) leafMass += e->w;

            // Passe 1 (gloutonne, O(n)) : gain immediat de chaque candidat.
            // on parcours toutes les features une par une 
            vector<pair<double,int>> greedyRanked;
            for (int f = 0; f < n; f++) {
                if (leaf.usedFeatures[f]) continue; // si déja utilisé on continue 
                double wL = 0, wPosL = 0, wR = 0, wPosR = 0;
                for (auto* e : leaf.examples) {
                    if (e->x[f] == 0) { wL += e->w; if (e->y == 1) wPosL += e->w; }
                    else              { wR += e->w; if (e->y == 1) wPosR += e->w; }
                }
                if (wL <= 0 || wR <= 0) continue;
                double imm = (wL / (wL + wR)) * computeGini(wL, wPosL)
                           + (wR / (wL + wR)) * computeGini(wR, wPosR);
                greedyRanked.push_back({ parentImpurity - imm, f }); // calcule le gain immédiat 
            }

            // on trie la liste des paires (gain, index_de_la_feature) par ordre decroisant 
            sort(greedyRanked.begin(), greedyRanked.end(), greater<pair<double,int>>());

            // Passe 2 (lookahead, O(min(n,LOOKAHEAD_TOPM) * n)) : raffine 
            // seulement les meilleurs candidats gloutons.
            int    leafBestFeature = -1;
            double leafBestGain    = -1.0;
            LookaheadBuffers buf(n);
            int limit = min((int)greedyRanked.size(), LOOKAHEAD_TOPM);

            // on teste les premieres de notre liste tries 
            for (int idx = 0; idx < limit; idx++) {
                int f = greedyRanked[idx].second;
                double wL, wR;
                // simulation de la 2 eme coupure pour chaque feature parmi celle choisi 
                double lookaheadImpurity = lookaheadImpurityForFeature(leaf.examples, leaf.usedFeatures, n, f, buf, wL, wR);
                if (lookaheadImpurity < 0) continue;
                double gain = parentImpurity - lookaheadImpurity;
                if (gain > leafBestGain) { leafBestGain = gain; leafBestFeature = f; }
            }

 


            // choix de la feuille qu'on va coupé : celle qui a une grande valeur gain*masse 
            double leafWeightedGain = leafBestFeature != -1 ? leafBestGain * leafMass : -1.0;
            if (leafBestFeature != -1 && leafWeightedGain > bestGain) {
                bestGain = leafWeightedGain; bestLeafIdx = (int)li; bestFeature = leafBestFeature;
            }
        }

        if (bestLeafIdx == -1) break; // si aucune feuille trouvée on sort de la boucle 

        // application de la coupure trouvé selon la bonne feature à la feuille mérité 
        auto& leaf = leaves[bestLeafIdx];
        RouterNode* node = leaf.node;
        node->isLeaf       = false;
        node->splitFeature = bestFeature;
        node->clauseIndex  = -1;
        node->left  = make_unique<RouterNode>(); //Création du fils Gauche
        node->right = make_unique<RouterNode>(); // Création du fils Droit
        node->left->isLeaf  = true; // Le fils gauche devient une feuille temporaire
        node->right->isLeaf = true; // Le fils droit devient une feuille temporaire

        // distribution des exemples 
        vector<const WeightedLabeledExample*> leftEx, rightEx;
        for (auto* e : leaf.examples) (e->x[bestFeature] == 0 ? leftEx : rightEx).push_back(e);

        // mise à jour de l'historique 
        vector<bool> usedLeft = leaf.usedFeatures, usedRight = leaf.usedFeatures;
        usedLeft[bestFeature] = usedRight[bestFeature] = true;
        vector<int> fixedLeft = leaf.fixedValue, fixedRight = leaf.fixedValue;
        fixedLeft[bestFeature] = 0; fixedRight[bestFeature] = 1;

        WLeafInfo newLeft  { node->left.get(),  leftEx,  usedLeft,  fixedLeft  };
        WLeafInfo newRight { node->right.get(), rightEx, usedRight, fixedRight };

        leaves.erase(leaves.begin() + bestLeafIdx);
        leaves.push_back(newLeft);
        leaves.push_back(newRight);
    }



    // remplir les tableauax de routeur 
   
    tree->leafUsedFeatures.resize(leaves.size());
    tree->leafFixedValue.resize(leaves.size());
    tree->leafIsPureZero.resize(leaves.size());
    tree->leafMajorityVote.resize(leaves.size());
    tree->leafWorstSkew.resize(leaves.size());


    for (size_t i = 0; i < leaves.size(); i++) {
        leaves[i].node->clauseIndex = (int)i;
        tree->leafUsedFeatures[i] = leaves[i].usedFeatures;
        tree->leafFixedValue[i]   = leaves[i].fixedValue;

        double wSum = 0, wPos = 0;
        for (auto* e : leaves[i].examples) { wSum += e->w; if (e->y == 1) wPos += e->w; }
        double frac1 = wSum > 0 ? wPos / wSum : 0.0;
        tree->leafIsPureZero[i] = frac1 <= noiseTol;
        tree->leafMajorityVote[i] = frac1 >= 0.5 ? 1 : 0;

        double worstSkew = 0.5;
        for (int f = 0; f < n; f++) {
            if (leaves[i].usedFeatures[f]) continue;
            double wf = 0;
            for (auto* e : leaves[i].examples) wf += e->w * e->x[f];
            double q = wSum > 0 ? wf / wSum : 0.5;
            worstSkew = min(worstSkew, min(q, 1.0 - q));
        }
        tree->leafWorstSkew[i] = worstSkew;
    }

    return tree;
}
