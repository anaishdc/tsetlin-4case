#include "tsetlin_engine_4case.h"
#include <cmath>
#include <iomanip>

// ===================================================================
// Validation empirique : a S=1 fixe, quel alpha choisir pour chaque
// niveau de bruit b = noiseProb ?
//
//   c        = P(x=1)
//   noiseProb= b = P(y=1|x=0)    ; a = P(y=1|x=1) = 1-noiseProb
//   S        = 1 (fixe)
//   alpha    = parametre balaye
//
//=====================================================================

struct Theory {
    double rhoV, rhoVbar, piV_Inc, piVbar_Exc, delta, alphaStar;
};

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
    double alphaStar = (delta > 0 && b > 0) ? S * delta / ((1 - c) * b) : -1.0;

    return { rhoV, rhoVbar, piV, piVbar, delta, alphaStar };
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
        DetailedMetrics m = runTrialDetailed(3000 + run, noiseProb, p, N, totalExamples, window);
        sumExact += m.exactRate; sumVInc += m.vIncludeRate; sumVbarExc += m.vbarExcludeRate;
    }
    return { sumExact / nbRuns, sumVInc / nbRuns, sumVbarExc / nbRuns };
}

int main() {
    const double c = 0.5;              // x genere uniforme -> P(x=1)=0.5
    const double S = 1.0;              // S fixe
    const int    N = 50;
    const long long total  = 3'000'000;
    const long long window = 100'000;
    const int       nbRuns = 20;

    cout << fixed << setprecision(4);

    // =========================================================
    // Courbe theorique alpha*(bruit), grille fine (instantane, pas de simulation)
    // =========================================================
    {
        ofstream f("test_D_alpha_seuils.csv");
        f << "noiseProb,alphaStar_theorie\n";
        for (double b = 0.02; b <= 0.48 + 1e-9; b += 0.02) {
            Theory th = computeTheory(c, b, 1.0, S, N);
            f << b << "," << th.alphaStar << "\n";
        }
        cout << "-> test_D_alpha_seuils.csv ecrit\n";
    }

    // =========================================================
    // Balayage empirique de alpha, a quelques niveaux de bruit fixes
    // =========================================================
    {
        vector<double> noises = {0.25, 0.30, 0.35, 0.40, 0.45};
        vector<double> alphas = {0.05,0.10,0.15,0.20,0.30,0.40,0.50,0.60,0.70,0.85,1.00};

        ofstream f("test_D_alpha_par_bruit.csv");
        f << "noiseProb,alpha,alphaStar_theorie,rhoV_theorie,piV_Inc_theorie,"
             "vIncludeRate_empirique,exactRate_empirique\n";

        for (double noise : noises) {
            double alphaStar = computeTheory(c, noise, 1.0, S, N).alphaStar;
            cout << "\n=== bruit=" << noise << "  alpha*_theorie=" << alphaStar << " ===\n";
            for (double alpha : alphas) {
                Params p = { S, alpha };
                Theory th = computeTheory(c, noise, alpha, S, N);
                DetailedMetrics m = runBatchDetailed(noise, p, N, total, window, nbRuns);
                f << noise << "," << alpha << "," << alphaStar << "," << th.rhoV << ","
                  << th.piV_Inc << "," << m.vIncludeRate << "," << m.exactRate << "\n";
                cout << "alpha=" << alpha << "  rhoV_theo=" << th.rhoV
                     << "  vInclude_empirique=" << m.vIncludeRate
                     << "  exact_empirique=" << m.exactRate << "\n";
            }
        }
    }

    cout << "\n-> CSV ecrits : test_D_alpha_seuils.csv, test_D_alpha_par_bruit.csv\n";
    return 0;
}
