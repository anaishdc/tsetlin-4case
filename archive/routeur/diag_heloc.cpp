#include "../tsetlin_engine.h"
#include <fstream>
#include <sstream>
#include <iostream>

vector<LabeledExample> loadDataset(const string& path, int n) {
    vector<LabeledExample> data;
    ifstream f(path); string line;
    while (getline(f, line)) {
        if (line.empty()) continue;
        istringstream iss(line);
        vector<int> x(n);
        for (int i = 0; i < n; i++) iss >> x[i];
        int y; iss >> y;
        data.push_back({x, y});
    }
    return data;
}

int main() {
    int n = 88;
    auto data = loadDataset("../data/BinaryhelocData.txt", n);
    cout << "n exemples: " << data.size() << "\n";

    rng.seed(20000);
    vector<WeightedLabeledExample> batch;
    double w = 1.0 / data.size();
    // reproduit discoverCount = max(300, 3*88=264) = 300, tri par poids (tous egaux ici)
    for (int i = 0; i < 300 && i < (int)data.size(); i++)
        batch.push_back({data[i].x, data[i].y, w});

    auto router = buildWeightedRouterTree(batch, n, 2, 0.10);
    cout << "K obtenu: " << router->leafUsedFeatures.size() << "\n";
    for (int i = 0; i < n; i++)
        if (router->leafUsedFeatures[0][i]) cout << "feature racine choisie par NOTRE routeur: x" << i << "\n";

    // purete des feuilles
    for (size_t k = 0; k < router->leafMajorityVote.size(); k++)
        cout << "feuille " << k << " majorityVote=" << router->leafMajorityVote[k]
             << " worstSkew=" << router->leafWorstSkew[k] << "\n";
    return 0;
}
