#!/usr/bin/env python3
# -----------------------------------------------------------------
# One-hot encode Monk-1/2/3 (UCI) : 6 attributs categoriels de
# cardinalites [3,3,2,3,4,2] -> 17 bits. Splits train/test OFFICIELS
# et FIXES (comme fournis par UCI), pas de reshuffle -- meme
# convention que XOR/Digits.
# -----------------------------------------------------------------
CARDS = [3, 3, 2, 3, 4, 2]  # a1..a6

def onehot_row(attrs):
    bits = []
    for val, card in zip(attrs, CARDS):
        oh = [0] * card
        oh[val - 1] = 1
        bits.extend(oh)
    return bits

def convert(src_path, dst_path):
    with open(src_path) as fin, open(dst_path, "w") as fout:
        for line in fin:
            parts = line.split()
            if not parts:
                continue
            label = int(parts[0])
            attrs = [int(v) for v in parts[1:7]]
            bits = onehot_row(attrs)
            fout.write(" ".join(str(b) for b in bits) + f" {label}\n")

if __name__ == "__main__":
    for i in [1, 2, 3]:
        convert(f"monks-{i}.train", f"../data/Monk{i}TrainingData.txt")
        convert(f"monks-{i}.test",  f"../data/Monk{i}TestData.txt")
        print(f"Monk-{i} : ecrit")
