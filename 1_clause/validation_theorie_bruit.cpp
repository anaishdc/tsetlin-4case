#include "tsetlin_engine_4case.h"
#include <cmath>
#include <iomanip>

// ===================================================================
// Validation empirique decas IDENTITY bruite, n=1
//
//   c        = P(x=1)           
//   noiseProb= b = P(y=1|x=0)    ; a = P(y=1|x=1) = 1-noiseProb
//   alpha    = parametre de la regle 
//   S        = parametre de la regle 
//
// Trois tests :
//   A. seuil sur S 
//   B. effet du bruit a S=1 fixe (limite de bruit tolerable)
//   C. effet de N (etats de l'automate) sur l'erreur de v isolement
// ===================================================================

struct Theory {
    double rhoV, rhoVbar, piV_Inc, piVbar_Exc, delta, Sstar;
};

// Calcule les predictions theoriques pour v et vbar (cas IDENTITY, n=1)
Theory computeTheory(double c, double noiseProb, double alpha, double S, int N) {
    double a = 1.0 - noiseProb;   // P(y=1|x=1)
    double b = noiseProb;         // P(y=1|x=0)

    double pPlusV     = (1 - c) * (1 - b) * S; 
    double pMinusV    = (1 - c) * b * alpha + c * (1 - a) * S;
    double pPlusVbar  = c * (1 - a) * S;
    double pMinusVbar = (1 - c) * (1 - b) * S + c * a * alpha;

    double rhoV    = pPlusV / pMinusV;
    double rhoVbar = pPlusVbar / pMinusVbar;

    double piV    = pow(rhoV, N) / (pow(rhoV, N) + 1.0);
    double piVbar = 1.0 / (pow(rhoVbar, N) + 1.0);

    double delta = (1 - c) * (1 - b) - c * (1 - a);
    double Sstar = (delta > 0) ? (1 - c) * b * alpha / delta : -1.0;

    return { rhoV, rhoVbar, piV, piVbar, delta, Sstar };
}


// ---------------------------------------------------------------
struct DetailedMetrics {
    double exactRate = 0, vIncludeRate = 0, vbarExcludeRate = 0;
};

DetailedMetrics runTrialDetailed(unsigned seed, double noiseProb,
                                  const Params& p, int N,
                                  long long totalExamples, long long window) {
    rng.seed(seed);
    const int n = 1;
    Clause clause(n, N);
    LiteralSet ls{ {true}, {false} };   // v doit inclure x1, vbar doit exclure ~x1
    uniform_int_distribution<int> bit(0, 1);

    long long vIncCount = 0, vbarExcCount = 0, exactCount = 0;
    for (long long t = 1; t <= totalExamples; t++) {
        vector<int> x(1);
        x[0] = bit(rng);
        int yTrue  = x[0];
        int yNoisy = flip(noiseProb) ? (1 - yTrue) : yTrue;
        clause.update(x, yNoisy, p);
        if (t > totalExamples - window) {
            if (clause.v[0].included())      vIncCount++;
            if (!clause.vbar[0].included())  vbarExcCount++;
            if (clause.isStructurallyExact(ls)) exactCount++;
        }
    }
    return { exactCount / (double)window, vIncCount / (double)window, vbarExcCount / (double)window };
}

DetailedMetrics runBatchDetailed(double noiseProb, const Params& p, int N,
                                  long long totalExamples, long long window, int nbRuns) {
    double sumExact = 0, sumVInc = 0, sumVbarExc = 0;
    for (int run = 0; run < nbRuns; run++) {
        DetailedMetrics m = runTrialDetailed(2000 + run, noiseProb, p, N, totalExamples, window);
        sumExact += m.exactRate; sumVInc += m.vIncludeRate; sumVbarExc += m.vbarExcludeRate;
    }
    return { sumExact / nbRuns, sumVInc / nbRuns, sumVbarExc / nbRuns };
}

int main() {
    const double c = 0.5;              // x genere uniforme -> P(x=1)=0.5
    const long long total  = 3'000'000;
    const long long window = 100'000;
    const int       nbRuns = 20;

    cout << fixed << setprecision(4);

    // =========================================================
    // TEST A : seuil sur S  (noiseProb=0.3, alpha=1 fixes)
    // =========================================================
    {
        const double noiseProb = 0.3, alpha = 1.0;
        const int N = 500;
        cout << "\n=== TEST A : seuil sur S (noiseProb=0.3, alpha=1, N=500) ===\n";
        double Sstar = computeTheory(c, noiseProb, alpha, 1.0, N).Sstar;
        cout << "Seuil theorique S* = " << Sstar << "\n\n";

        ofstream f("test_A_seuil_S.csv");
        f << "S,rhoV_theorie,piV_Inc_theorie,vIncludeRate_empirique,exactRate_empirique\n";
        vector<double> Svals = {0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.65,
                                 0.70,0.75,0.80,0.85,0.90,1.00};
        for (double S : Svals) {
            Params p = { S, S, S, S, alpha, alpha };
            Theory th = computeTheory(c, noiseProb, alpha, S, N);
            DetailedMetrics m = runBatchDetailed(noiseProb, p, N, total, window, nbRuns);
            f << S << "," << th.rhoV << "," << th.piV_Inc << ","
              << m.vIncludeRate << "," << m.exactRate << "\n";
            cout << "S=" << S << "  rhoV_theo=" << th.rhoV
                 << "  piV_theo=" << th.piV_Inc
                 << "  vInclude_empirique=" << m.vIncludeRate
                 << "  exact_empirique=" << m.exactRate << "\n";
        }
    }

    // =========================================================
    // TEST B : effet du bruit a S=1 fixe (alpha=1 fixe)
    // =========================================================
    {
        const double S = 1.0, alpha = 1.0;
        const int N = 50;
        cout << "\n=== TEST B : effet du bruit (S=1, alpha=1, N=50) ===\n";

        ofstream f("test_B_effet_bruit.csv");
        f << "noiseProb,Sstar_theorie,rhoV_theorie,piV_Inc_theorie,"
             "vIncludeRate_empirique,exactRate_empirique\n";
        vector<double> noises = {0.05,0.10,0.15,0.20,0.25,0.30,0.32,
                                  0.333,0.34,0.36,0.38,0.40,0.45};
        for (double noise : noises) {
            Params p = { S, S, S, S, alpha, alpha };
            Theory th = computeTheory(c, noise, alpha, S, N);
            DetailedMetrics m = runBatchDetailed(noise, p, N, total, window, nbRuns);
            f << noise << "," << th.Sstar << "," << th.rhoV << "," << th.piV_Inc << ","
              << m.vIncludeRate << "," << m.exactRate << "\n";
            cout << "noise=" << noise << "  S*_theo=" << th.Sstar
                 << "  rhoV_theo=" << th.rhoV
                 << "  vInclude_empirique=" << m.vIncludeRate
                 << "  exact_empirique=" << m.exactRate << "\n";
        }
    }

    // =========================================================
    // TEST C : effet de N (S=1, noiseProb=0.3, alpha=1)
    // =========================================================
    {
        const double S = 1.0, noiseProb = 0.3, alpha = 1.0;
        cout << "\n=== TEST C : effet de N (S=1, noiseProb=0.3, alpha=1) ===\n";

        ofstream f("test_C_effet_N.csv");
        f << "N,rhoV_theorie,piV_Inc_theorie,erreur_theorie,"
             "vIncludeRate_empirique,erreur_empirique\n";
        vector<int> Nvals = {2,3,5,8,11,15,20,25,30,40,60};
        for (int N : Nvals) {
            Params p = { S, S, S, S, alpha, alpha };
            Theory th = computeTheory(c, noiseProb, alpha, S, N);
            long long thisTotal = max(total, (long long)N * 200'000LL);
            DetailedMetrics m = runBatchDetailed(noiseProb, p, N, thisTotal, window, nbRuns);
            double errTheo = 1.0 - th.piV_Inc;
            double errEmp  = 1.0 - m.vIncludeRate;
            f << N << "," << th.rhoV << "," << th.piV_Inc << "," << errTheo << ","
              << m.vIncludeRate << "," << errEmp << "\n";
            cout << "N=" << N << "  erreur_theo=" << errTheo
                 << "  erreur_empirique=" << errEmp << "\n";
        }
    }

    cout << "\n-> CSV ecrits : test_A_seuil_S.csv, test_B_effet_bruit.csv, test_C_effet_N.csv\n";
    return 0;
}
