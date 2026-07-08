#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
using namespace std;

// Extension de fourier_detect.cpp : ajoute le degre 4 (necessaire
// pour le multiplexeur 11-bit : 3 bits d'adresse + 1 bit de donnee =
// interaction de degre 4). Teste aussi tous les degres 1-4.

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
    int maxDegree = argc > 3 ? atoi(argv[3]) : 4;

    vector<int> y;
    auto X = loadX(path, n, y);
    int N = (int)X.size();

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

    if (maxDegree >= 1) for (int i = 0; i < n; i++) top.push_back({evalSubset({i}), {i}});
    if (maxDegree >= 2)
        for (int i = 0; i < n; i++)
            for (int j = i+1; j < n; j++)
                top.push_back({evalSubset({i,j}), {i,j}});
    if (maxDegree >= 3)
        for (int i = 0; i < n; i++)
            for (int j = i+1; j < n; j++)
                for (int k = j+1; k < n; k++)
                    top.push_back({evalSubset({i,j,k}), {i,j,k}});
    if (maxDegree >= 4)
        for (int i = 0; i < n; i++)
            for (int j = i+1; j < n; j++)
                for (int k = j+1; k < n; k++)
                    for (int l = k+1; l < n; l++)
                        top.push_back({evalSubset({i,j,k,l}), {i,j,k,l}});

    sort(top.begin(), top.end(), [](const Coef& a, const Coef& b) { return fabs(a.val) > fabs(b.val); });

    cout << "Top 15 coefficients de Fourier (degre <= " << maxDegree << ", |valeur| decroissante) :\n";
    for (int i = 0; i < min(15, (int)top.size()); i++) {
        cout << "  S={";
        for (size_t k = 0; k < top[i].subset.size(); k++) cout << (k?",":"") << top[i].subset[k];
        cout << "}  f_hat=" << top[i].val << "\n";
    }
    return 0;
}
