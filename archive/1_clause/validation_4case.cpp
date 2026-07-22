#include "tsetlin_engine_4case.h"

// ---------------------------------------------------------------
// Configs de validation : (n, k, nom, target, LiteralSet)
// n : nombre de variables X =(x1,x2,...,xn)
// k : nombre de littéraux inclu dans la clause cible 
// nom : la formule de la clause cible 
// target : fonction qui calcule la valeur d'une clause
// LiteralSet : la reponses attendu : quels automates doivent finir en include 
// ---------------------------------------------------------------
struct Config {
    int        n, k;
    string     name;
    Target     target;
    LiteralSet ls;
};



vector<Config> makeConfigs() {
    return {
        { 2, 1, "n=2 k=1  x1",
          [](const vector<int>& x) { return x[0]; },
          { {true,  false},
            {false, false} }
        },
        { 2, 1, "n=2 k=1  ¬x1",
          [](const vector<int>& x) { return x[0] == 0 ? 1 : 0; },
          { {false, false},
            {true,  false} }
        },
        { 2, 2, "n=2 k=2  x1^¬x2",
          [](const vector<int>& x) { return (x[0]==1 && x[1]==0) ? 1 : 0; },
          { {true,  false},
            {false, true } }
        },
        { 4, 2, "n=4 k=2  x1^¬x2",
          [](const vector<int>& x) { return (x[0]==1 && x[1]==0) ? 1 : 0; },
          { {true,  false, false, false},
            {false, true,  false, false} }
        },
        { 4, 3, "n=4 k=3  x1^¬x2^x3",
          [](const vector<int>& x) { return (x[0]==1 && x[1]==0 && x[2]==1) ? 1 : 0; },
          { {true,  false, true,  false},
            {false, true,  false, false} }
        },
        { 6, 3, "n=6 k=3  x1^¬x2^x5",
          [](const vector<int>& x) { return (x[0]==1 && x[1]==0 && x[4]==1) ? 1 : 0; },
          { {true,  false, false,  false, true, false},
            {false, true,  false, false, false, false} }
        },
    };
}

// ---------------------------------------------------------------
// Affichage : séparateur de table
// ---------------------------------------------------------------
void ligne(const string& gauche, const string& milieu, const string& droite,
           const vector<int>& largeurs) {
    cout << gauche;
    for (int i = 0; i < (int)largeurs.size(); i++) {
        cout << string(largeurs[i] + 2, '-');
        cout << (i + 1 < (int)largeurs.size() ? milieu : droite);
    }
    cout << "\n";
}

void entete(const vector<string>& cols, const vector<int>& larg) {
    ligne("+", "+", "+", larg);
    cout << "|";
    for (int i = 0; i < (int)cols.size(); i++)
        printf(" %-*s |", larg[i], cols[i].c_str());
    cout << "\n";
    ligne("+", "+", "+", larg);
}

void piedTable(const vector<int>& larg) { ligne("+", "+", "+", larg); }

// ---------------------------------------------------------------
// Imprime une ligne de données dans le tableau
// ---------------------------------------------------------------
void ligneData(const string& label, double exact, double stdE,
               double acc, double stdA, const vector<int>& larg) {
    cout << "|";
    printf(" %-*s |", larg[0], label.c_str());
    printf(" %*.1f%%  |", larg[1] - 3, exact);
    printf(" %*.1f%%  |", larg[2] - 3, stdE);
    printf(" %*.1f%%  |", larg[3] - 3, acc);
    printf(" %*.1f%%  |", larg[4] - 3, stdA);
    cout << "\n";
}

// Imprime une ligne avec en plus la clause apprise par la machine
void ligneDataClause(const string& label, double exact, double stdE,
                     double acc, double stdA, const string& clause,
                     const vector<int>& larg) {
    cout << "|";
    printf(" %-*s |", larg[0], label.c_str());
    printf(" %*.1f%%  |", larg[1] - 3, exact);
    printf(" %*.1f%%  |", larg[2] - 3, stdE);
    printf(" %*.1f%%  |", larg[3] - 3, acc);
    printf(" %*.1f%%  |", larg[4] - 3, stdA);
    printf(" %-*s |", larg[5], clause.c_str());
    cout << "\n";
}


// ==========================================================================
// SECTION 1 : validation sans bruit
// ==========================================================================

int main() {
    const Params p      = {1,0.3}; // S1=S, a1=a (regle a 4 cas) ; S2,S3,S4,a2 inutilises
    const int    total  = 100000;//  nombre total d'exemple
    const int    window = 4000; //evaluations sur les 4000 derniers exemples 
    const int    nbRuns = 50; // nb simulations 

    const int    N      = 320; // nombre d'etat de la moitié de l'automate 

    auto configs = makeConfigs();

    vector<int> larg = { 22, 8, 6, 8, 6 };
    vector<int> larg1 = { 22, 8, 6, 8, 6, 18 };

    // ==========================================================================
    // SECTION 1 : validation sans bruit
    // ==========================================================================
    cout << "\n";
    cout << "================================================================\n";
    cout << "  SECTION 1 : Validation sans bruit, moteur 4 cas (epsilon=0%, " << nbRuns << " runs)\n";
    cout << "================================================================\n\n";

    entete({"config", "exact", "+/-", "accuracy", "+/-", "clause trouvee"}, larg1);

    vector<Metrics> res1;
    for (auto& cfg : configs) {
        TestSet ts = makeTestSet(cfg.n, cfg.target);// creer un testset surlequel on evalue notre machine 
        Metrics m  = runBatch(0.0, cfg.n, cfg.target, cfg.ls, ts, total, window, nbRuns, p, N); // lancer les simulations 
        res1.push_back(m);
        string clause = learnedClauseStr(1000, cfg.n, cfg.target, total, 0.0, p, N);// afficher la derniere clause apprise 
        ligneDataClause(cfg.name, m.exactRate*100, m.stdExactRate*100,
                        m.accuracy*100, m.stdAccuracy*100, clause, larg1);
    }
    piedTable(larg1);

    

    // ==========================================================================
    // SECTION 2 : robustesse au bruit 
    // ==========================================================================
    cout << "\n\n";
    cout << "================================================================\n";
    cout << "  SECTION 2 : Effet du bruit par config, moteur 4 cas\n";
    cout << "================================================================\n";

    vector<int> larg2 = { 8, 8, 6, 8, 6, 18 };

    for (auto& cfg : configs) {
        cout << "\n  Config : " << cfg.name << "\n";
        entete({"bruit", "exact", "+/-", "accuracy", "+/-", "clause trouvee"}, larg2);

        TestSet ts = makeTestSet(cfg.n, cfg.target);
        vector<pair<double, Metrics>> rows;

        for (double noise : { 0.00, 0.05, 0.10, 0.20, 0.30, 0.40 }) {
            Metrics m = runBatch(noise, cfg.n, cfg.target, cfg.ls, ts, total, window, nbRuns, p, N);
            string label = to_string((int)round(noise*100)) + "%";
            string clause = learnedClauseStr(1000, cfg.n, cfg.target, total, noise, p, N);
            ligneDataClause(label, m.exactRate*100, m.stdExactRate*100,
                            m.accuracy*100, m.stdAccuracy*100, clause, larg2);
            rows.emplace_back(noise, m);
        }
        piedTable(larg2);

        string fname = "val4case_n" + to_string(cfg.n) + "_k" + to_string(cfg.k) + ".csv";
        writeCsv(fname, "bruit", rows);
    }

    return 0;
}
