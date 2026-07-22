#!/usr/bin/env python3
# -----------------------------------------------------------------
# Binarisation thermometre (meme principe que Iris/Digits) pour des
# datasets UCI a features continues : Breast Cancer Wisconsin, Wine.
# 4 bits/feature (comme Iris), seuils = quartiles empiriques du
# dataset entier (25/50/75e percentiles).
# -----------------------------------------------------------------
import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine

def thermometer_encode(X, bits_per_feature=4):
    n, d = X.shape
    out = np.zeros((n, d * bits_per_feature), dtype=int)
    for f in range(d):
        col = X[:, f]
        qs = np.percentile(col, np.linspace(0, 100, bits_per_feature + 2)[1:-1])
        for b, q in enumerate(qs):
            out[:, f * bits_per_feature + b] = (col > q).astype(int)
    return out

def write_dataset(path, Xbin, y):
    with open(path, "w") as fh:
        for i in range(Xbin.shape[0]):
            fh.write(" ".join(str(v) for v in Xbin[i]) + f" {y[i]}\n")

if __name__ == "__main__":
    bc = load_breast_cancer()
    Xbc = thermometer_encode(bc.data, 4)
    print("Breast Cancer:", Xbc.shape, "classes:", set(bc.target.tolist()))
    write_dataset("../data/BinaryBreastCancerData.txt", Xbc, bc.target)

    wine = load_wine()
    Xw = thermometer_encode(wine.data, 4)
    print("Wine:", Xw.shape, "classes:", set(wine.target.tolist()))
    write_dataset("../data/BinaryWineData.txt", Xw, wine.target)

    print("Ecrit : data/BinaryBreastCancerData.txt, data/BinaryWineData.txt")
