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

    for (int dc : {300, 1000, 3000, 8000}) {
        rng.seed(20000);
        vector<WeightedLabeledExample> batch;
        double w = 1.0 / data.size();
        for (int i = 0; i < dc && i < (int)data.size(); i++)
            batch.push_back({data[i].x, data[i].y, w});

        auto router = buildWeightedRouterTree(batch, n, 2, 0.10);
        cout << "discoverCount=" << dc << " -> feature racine: ";
        for (int i = 0; i < n; i++)
            if (router->leafUsedFeatures[0][i]) cout << "x" << i;
        cout << "  votes=[" << router->leafMajorityVote[0] << "," << router->leafMajorityVote[1] << "]\n";
    }
    return 0;
}
