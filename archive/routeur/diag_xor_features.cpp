#include "../tsetlin_engine.h"
#include <fstream>
#include <sstream>
#include <iostream>

// Diagnostic : quelles features le routeur utilise-t-il (donc gele)
// a differents K, sur XOR ? Verifie si x0/x1 (les 2 features
// informatives) se font geler quand K augmente.

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
    int n = 12;
    auto trainData = loadDataset("../data/NoisyXORTrainingData.txt", n);

    for (int K : {2, 4, 8}) {
        rng.seed(30000);
        vector<WeightedLabeledExample> batch;
        double w = 1.0 / trainData.size();
        for (auto& e : trainData) batch.push_back({e.x, e.y, w});
        auto router = buildWeightedRouterTree(batch, n, K, 0.40);

        vector<bool> usedAny(n, false);
        for (auto& leafFeatures : router->leafUsedFeatures)
            for (int i = 0; i < n; i++) if (leafFeatures[i]) usedAny[i] = true;

        cout << "K=" << K << " (" << router->leafUsedFeatures.size() << " feuilles) -- features utilisees/gelees : ";
        for (int i = 0; i < n; i++) if (usedAny[i]) cout << "x" << i << " ";
        cout << "\n  -> x0/x1 (informatives) gelees ? "
             << (usedAny[0] || usedAny[1] ? "OUI" : "non") << "\n";
    }
    return 0;
}
