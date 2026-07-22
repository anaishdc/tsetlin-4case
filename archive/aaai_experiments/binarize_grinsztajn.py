#!/usr/bin/env python3
# -----------------------------------------------------------------
# Parse + binarise (thermometre, 4 bits/feature) les 6 datasets du
# benchmark Grinsztajn et al. 2022 (suite OpenML 337, classification
# sur features numeriques -- deja pre-filtrees, pas de categoriel).
# -----------------------------------------------------------------
import numpy as np

DATASETS = {
    "bank-marketing":  ("grinsztajn_raw/bank-marketing.arff", "Class"),
    "MagicTelescope":  ("grinsztajn_raw/MagicTelescope.arff", "class"),
    "credit":          ("grinsztajn_raw/credit.arff", "SeriousDlqin2yrs"),
    "house_16H":       ("grinsztajn_raw/house_16H.arff", "binaryClass"),
    "eye_movements":   ("grinsztajn_raw/eye_movements.arff", "label"),
    "heloc":           ("grinsztajn_raw/heloc.arff", "RiskPerformance"),
}

def parse_arff(path):
    attrs = []
    rows = []
    in_data = False
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            if line.upper().startswith("@ATTRIBUTE"):
                attrs.append(line.split()[1])
                continue
            if line.upper().startswith("@DATA"):
                in_data = True
                continue
            if line.upper().startswith("@RELATION"):
                continue
            if in_data:
                rows.append(line.split(","))
    return attrs, rows

def thermometer_encode(X, bits_per_feature=4):
    n, d = X.shape
    out = np.zeros((n, d * bits_per_feature), dtype=int)
    for f in range(d):
        col = X[:, f]
        qs = np.percentile(col, np.linspace(0, 100, bits_per_feature + 2)[1:-1])
        for b, q in enumerate(qs):
            out[:, f * bits_per_feature + b] = (col > q).astype(int)
    return out

if __name__ == "__main__":
    for name, (path, label_col) in DATASETS.items():
        attrs, rows = parse_arff(path)
        label_idx = attrs.index(label_col)
        feat_idx = [i for i in range(len(attrs)) if i != label_idx]

        X = np.array([[float(r[i]) for i in feat_idx] for r in rows])
        raw_labels = [r[label_idx] for r in rows]
        classes = sorted(set(raw_labels))
        assert len(classes) == 2, f"{name}: attendu binaire, trouve {classes}"
        y = np.array([0 if v == classes[0] else 1 for v in raw_labels])

        Xbin = thermometer_encode(X, 4)
        out_path = f"../data/Binary{name.replace('-', '').replace('_','')}Data.txt"
        with open(out_path, "w") as fh:
            for i in range(Xbin.shape[0]):
                fh.write(" ".join(str(v) for v in Xbin[i]) + f" {y[i]}\n")
        print(f"{name}: {X.shape[0]} exemples, {X.shape[1]} features -> {Xbin.shape[1]} bits, "
              f"classes={classes} -> {out_path}")
