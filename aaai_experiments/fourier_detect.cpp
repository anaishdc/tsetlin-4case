#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
using namespace std;

// Detection des sous-ensembles de variables pertinents via les
// coefficients de Fourier empiriques (encodage +-1) :
//   f_hat(S) = E[ y_signe * produit_{i in S}(x_i_signe) ]
// Pour une parite pure a k bits, TOUTE l'energie est concentree sur
// le coefficient du bon sous-ensemble S (|f_hat(S)| ~ 1), tous les
// autres sont ~0 (bruit d'echantillonnage). Teste tous les
// sous-ensembles de taille 1, 2, 3.

vector<vector<int>> loadX(const string& path, int n, vector<int>& y) {
    vector<vector<int>> X;
    ifstream f(path); string line;
    while (getline(f, line)) {
        if (line.empty()) continue;
        istringstream iss(line);
        vector<int> x(n);
        for (int i = 0; i < n; i++) iss >> x[i];
        int label; iss >> label;
        X.push_back(x); y.push_back(label);
    }
    return X;
}

int main(int argc, char** argv) {
    string path = argv[1];
    int n = atoi(argv[2]);
    int maxDegree = argc > 3 ? atoi(argv[3]) : 3;

    vector<int> y;
    auto X = loadX(path, n, y);
    int N = (int)X.size();

    // encodage +-1
    vector<int> ys(N);
    for (int i = 0; i < N; i++) ys[i] = 2 * y[i] - 1;
    vector<vector<int>> xs(N, vector<int>(n));
    for (int i = 0; i < N; i++) for (int j = 0; j < n; j++) xs[i][j] = 2 * X[i][j] - 1;

    struct Coef { double val; vector<int> subset; };
    vector<Coef> top;

    auto evalSubset = [&](const vector<int>& subset) {
        double sum = 0;
        for (int i = 0; i < N; i++) {
            int prod = ys[i];
            for (int j : subset) prod *= xs[i][j];
            sum += prod;
        }
        return sum / N;
    };

    // degre 1
    for (int i = 0; i < n; i++) {
        double c = evalSubset({i});
        top.push_back({c, {i}});
    }
    // degre 2
    if (maxDegree >= 2) {
        for (int i = 0; i < n; i++)
            for (int j = i+1; j < n; j++) {
                double c = evalSubset({i,j});
                top.push_back({c, {i,j}});
            }
    }
    // degre 3
    if (maxDegree >= 3) {
        for (int i = 0; i < n; i++)
            for (int j = i+1; j < n; j++)
                for (int k = j+1; k < n; k++) {
                    double c = evalSubset({i,j,k});
                    top.push_back({c, {i,j,k}});
                }
    }

    sort(top.begin(), top.end(), [](const Coef& a, const Coef& b) { return fabs(a.val) > fabs(b.val); });

    cout << "Top 10 coefficients de Fourier (|valeur| decroissante) :\n";
    for (int i = 0; i < min(10, (int)top.size()); i++) {
        cout << "  S={";
        for (size_t k = 0; k < top[i].subset.size(); k++) cout << (k?",":"") << top[i].subset[k];
        cout << "}  f_hat=" << top[i].val << "\n";
    }
    return 0;
}
