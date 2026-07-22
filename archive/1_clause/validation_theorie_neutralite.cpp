#include "tsetlin_engine_4case.h"
#include <cmath>
#include <iomanip>

// ===================================================================
// Validation empirique du Theoreme "Neutralite des litteraux non
// pertinents" (convergence_clause_conditionnee.tex, Section 4).
//
// n=2 features, TOUTES LIBRES (pas de conditionnement K ici -- on teste
// directement la dynamique multi-features d'une seule clause) :
//   - feature 0 : pertinente, IDENTITY bruite (c0=0.5, a=0.7, b=0.3),
//     comme le Test A du document precedent.
//   - feature 1 : NON pertinente, independante de y, mais avec une
//     marginale c1 = P(x1=1) potentiellement desequilibree (!=0.5).
//
// p_y = P(y=1) = c0*a + (1-c0)*b = 0.5 (avec c0=0.5,a=0.7,b=0.3).
// Prediction theorique (Theoreme 4.2, cas c1>0.5, p_y=0.5) :
//   S_1^max = c1 * alpha / (2*c1 - 1)
//
// Deux tests :
//   D. seuil sur S a c1 fixe (desequilibre marginal fort)
//   E. seuil sur c1 a S fixe (a partir de quel desequilibre ca casse)
// ===================================================================

struct TheoryNeutre {
    double rho_v1, rho_vbar1, S1max;
};

// p_y et c0/a/b fixes par le test A (voir plus haut) -> p_y=0.5
TheoryNeutre computeTheoryNeutre(double c1, double alpha, double S, double p_y) {
    double pPlusV1     = (1 - c1) * (1 - p_y) * S;
    double pMinusV1    = (1 - c1) * p_y * alpha + c1 * (1 - p_y) * S;
    double pPlusVbar1  = c1 * (1 - p_y) * S;
    double pMinusVbar1 = (1 - c1) * (1 - p_y) * S + c1 * p_y * alpha;

    double rhoV1    = pPlusV1 / pMinusV1;
    double rhoVbar1 = pPlusVbar1 / pMinusVbar1;

    // Seuil S_1^max : formule fermee (cas c1>0.5 utilise ici)
    double S1max = (c1 > 0.5)
        ? c1 * p_y * alpha / ((2 * c1 - 1) * (1 - p_y))
        : (1 - c1) * p_y * alpha / ((1 - 2 * c1) * (1 - p_y));

    return { rhoV1, rhoVbar1, S1max };
}

struct NeutraliteMetrics {
    double v0IncRate = 0, vbar0ExcRate = 0;   // feature pertinente : doit converger vers (Inc, Exc)
    double v1IncRate = 0, vbar1IncRate = 0;   // feature non pertinente : les DEUX doivent rester Exclude
    double exactRate = 0;                      // les deux features dans la bonne config simultanement
};

NeutraliteMetrics runTrial(unsigned seed, double c0, double noiseProb, double c1,
                  const Params& p, int N, long long totalExamples, long long window) {
    rng.seed(seed);
    const int n = 2;
    Clause clause(n, N);
    LiteralSet ls{ {true, false}, {false, false} };  // feature0: (Inc,Exc) ; feature1: (Exc,Exc)
    uniform_real_distribution<double> u(0.0, 1.0);

    long long v0Inc=0, vbar0Exc=0, v1Inc=0, vbar1Inc=0, exact=0;
    for (long long t = 1; t <= totalExamples; t++) {
        vector<int> x(2);
        x[0] = (u(rng) < c0) ? 1 : 0;
        int yTrue  = x[0];
        int yNoisy = flip(noiseProb) ? (1 - yTrue) : yTrue;
        x[1] = (u(rng) < c1) ? 1 : 0;   // independant de y, marginale c1
        clause.update(x, yNoisy, p);
        if (t > totalExamples - window) {
            if (clause.v[0].included())    v0Inc++;
            if (!clause.vbar[0].included()) vbar0Exc++;
            if (clause.v[1].included())    v1Inc++;
            if (clause.vbar[1].included()) vbar1Inc++;
            if (clause.isStructurallyExact(ls)) exact++;
        }
    }
    double w = (double)window;
    return { v0Inc/w, vbar0Exc/w, v1Inc/w, vbar1Inc/w, exact/w };
}

NeutraliteMetrics runBatch(double c0, double noiseProb, double c1, const Params& p, int N,
                  long long totalExamples, long long window, int nbRuns) {
    NeutraliteMetrics sum{};
    for (int run = 0; run < nbRuns; run++) {
        NeutraliteMetrics m = runTrial(3000 + run, c0, noiseProb, c1, p, N, totalExamples, window);
        sum.v0IncRate += m.v0IncRate; sum.vbar0ExcRate += m.vbar0ExcRate;
        sum.v1IncRate += m.v1IncRate; sum.vbar1IncRate += m.vbar1IncRate;
        sum.exactRate += m.exactRate;
    }
    sum.v0IncRate/=nbRuns; sum.vbar0ExcRate/=nbRuns;
    sum.v1IncRate/=nbRuns; sum.vbar1IncRate/=nbRuns; sum.exactRate/=nbRuns;
    return sum;
}

int main() {
    const double c0 = 0.5, noiseProb = 0.3;   // feature pertinente : identique au Test A
    const double p_y = 0.5;                    // c0*a + (1-c0)*b = 0.5*0.7+0.5*0.3
    const long long total  = 3'000'000;
    const long long window = 100'000;
    const int nbRuns = 20;

    cout << fixed << setprecision(4);

    // =========================================================
    // TEST D : seuil sur S, a c1 fixe (desequilibre marginal fort)
    // S et alpha sont des PROBABILITES, bornees dans (0,1] -- alpha=1
    // rend le seuil S_1^max > 1 (donc jamais atteignable) pour p_y=0.5 ;
    // on prend alpha=0.15 pour ramener le seuil dans la plage valide.
    // =========================================================
    {
        const double c1 = 0.8, alpha = 0.15;
        const int N = 50;
        cout << "\n=== TEST D : seuil sur S (c1=0.8, alpha=0.15, N=50) ===\n";
        double S1max = computeTheoryNeutre(c1, alpha, 1.0, p_y).S1max;
        cout << "Seuil theorique S_1^max = " << S1max << "\n\n";

        ofstream f("test_D_seuil_S_neutralite.csv");
        f << "S,rhoV1_theorie,v1IncRate_empirique,vbar1IncRate_empirique,"
             "v0IncRate_empirique,exactRate_empirique\n";
        vector<double> Svals = {0.02,0.05,0.08,0.10,0.12,0.15,0.18,0.20,
                                 0.22,0.25,0.30,0.40,0.60,1.00};
        for (double S : Svals) {
            Params p = { S, alpha };
            TheoryNeutre th = computeTheoryNeutre(c1, alpha, S, p_y);
            NeutraliteMetrics m = runBatch(c0, noiseProb, c1, p, N, total, window, nbRuns);
            f << S << "," << th.rho_v1 << "," << m.v1IncRate << "," << m.vbar1IncRate << ","
              << m.v0IncRate << "," << m.exactRate << "\n";
            cout << "S=" << S << "  rhoV1_theo=" << th.rho_v1
                 << "  v1Include_empirique=" << m.v1IncRate
                 << "  (predit spurious si S>" << S1max << ")"
                 << "  exact_empirique=" << m.exactRate << "\n";
        }
    }

    // =========================================================
    // TEST E : seuil sur c1, a S fixe (S=0.5, alpha=0.15 : le seuil
    // theorique franchit S=0.5 vers c1*=0.588, cf commentaire calcule
    // a la main : c1*alpha/(2c1-1) = S  =>  c1* = S/(2S-alpha) = 0.5/0.85
    // =========================================================
    {
        const double S = 0.5, alpha = 0.15;
        const int N = 50;
        cout << "\n=== TEST E : seuil sur c1 (S=0.5, alpha=0.15, N=50) ===\n";

        ofstream f("test_E_seuil_c1_neutralite.csv");
        f << "c1,S1max_theorie,v1IncRate_empirique,vbar1IncRate_empirique,exactRate_empirique\n";
        vector<double> c1vals = {0.50,0.52,0.55,0.58,0.60,0.62,0.65,0.70,0.75,0.80,0.85,0.90,0.95};
        for (double c1 : c1vals) {
            Params p = { S, alpha };
            TheoryNeutre th = computeTheoryNeutre(c1, alpha, S, p_y);
            NeutraliteMetrics m = runBatch(c0, noiseProb, c1, p, N, total, window, nbRuns);
            f << c1 << "," << th.S1max << "," << m.v1IncRate << "," << m.vbar1IncRate << ","
              << m.exactRate << "\n";
            cout << "c1=" << c1 << "  S_1^max_theo=" << th.S1max
                 << "  (S=1 " << (th.S1max > 1.0 ? "sous" : "AU-DESSUS DU") << " seuil)"
                 << "  v1Include_empirique=" << m.v1IncRate
                 << "  exact_empirique=" << m.exactRate << "\n";
        }
    }

    cout << "\n-> CSV ecrits : test_D_seuil_S_neutralite.csv, test_E_seuil_c1_neutralite.csv\n";
    return 0;
}
