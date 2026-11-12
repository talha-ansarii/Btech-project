"""Write the fixed random labelled subsets used by every experiment.

For each seed we shuffle the 795 training images once and take the first 5% (40 images)
and the first 10% (80 images). So the 10% set always contains the 5% set, which is what
an active-learning round looks like (you keep what you already labelled and add more).

Files: splits/random_frac0.05_seed0.json  ->  {"indices": [...], "fraction": 0.05, "seed": 0}
These files are committed, so everyone trains on exactly the same images.
"""
import json
import os

import numpy as np

N_TRAIN = 795
HERE = os.path.dirname(os.path.abspath(__file__))


def main(seeds=(0, 1, 2), fractions=(0.05, 0.10)):
    os.makedirs(os.path.join(HERE, "splits"), exist_ok=True)
    for seed in seeds:
        perm = np.random.default_rng(seed).permutation(N_TRAIN)
        for frac in fractions:
            k = int(round(frac * N_TRAIN))
            sel = sorted(int(i) for i in perm[:k])
            path = os.path.join(HERE, "splits", f"random_frac{frac:.2f}_seed{seed}.json")
            with open(path, "w") as fh:
                json.dump({"method": "random", "fraction": frac, "seed": seed, "n": k, "indices": sel}, fh)
            print(path, k)


if __name__ == "__main__":
    main()
