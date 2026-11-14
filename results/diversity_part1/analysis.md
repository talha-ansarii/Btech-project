# Diversity Part 1: results and significance (Kaggle T4 x2, 2026-11-14, commit a9cc4ba)

RGB-D, NYUv2 40 classes, 4000 iterations, seeds 0-2, final checkpoint mIoU on the 654 test images.

| labels | method | mIoU (mean ± std) | per seed (0, 1, 2) | vs random (Welch p) |
|---|---|---|---|---|
| 5% | random (earlier baseline runs) | 15.09 ± 0.99 | | |
| 5% | kmeans (priority 1, Colab) | 16.65 ± 0.63 | 17.04, 16.98, 15.92 | 0.10 |
| 5% | kmeans-rgbd | 16.67 ± 0.78 | 17.39, 16.79, 15.84 | 0.10 |
| 5% | **cover-rgbd** | **17.89 ± 0.78** | 18.18, 17.01, 18.48 | **0.02** |
| 10% | random (earlier baseline runs) | 20.77 ± 0.54 | | |
| 10% | kmeans | 20.90 ± 1.34 | 22.42, 20.35, 19.92 | 0.89 |
| 10% | kmeans-rgbd | 21.75 ± 0.79 | 21.12, 22.63, 21.49 | 0.16 |
| 10% | **cover-rgbd** | **22.77 ± 0.60** | 23.33, 22.83, 22.14 | **0.01** |

The random per-seed values are not in the repo, so these tests use the reported mean and std with n = 3.

## Reading

* cover-rgbd beats random at both budgets: +2.8 mIoU at 5% and +2.0 at 10%. Both are significant at p < 0.05,
  but with only 3 seeds per side. It is also the only method that beats random at 10%.
* Against plain kmeans the gain is +1.2 at 5% and +1.9 at 10%. It is positive on every seed pair but **not significant
  with 3 seeds** (paired p 0.23 at 5%, 0.06 at 10%). This is the comparison that needs more seeds.
* kmeans-rgbd equals kmeans at 5% and is a little higher at 10%, so most of the gain comes from region coverage and not
  from adding depth to the global features. cover-rgb (part 2) tests whether depth helps inside the coverage method.
* The class-coverage table barely separates the methods. At 5%, cover-rgbd has 39.3 classes present against 38.7 for
  random, and a class entropy of 0.793 against 0.783. So "covers more classes" is not yet the explanation. analyze_splits.py
  now also reports the share of labelled pixels in the 20 rarest classes, which part 2 prints.

## Caveats

* The random and kmeans 5% numbers came from earlier runs (Colab and single-GPU Kaggle). Part 2 reruns random in the
  same notebook so that every comparison uses the same setup.
* With two runs in parallel, a run took 36 to 40 minutes instead of about 26 (CPU data loading is shared).
