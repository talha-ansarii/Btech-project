# Diversity Parts 1 + 2: all methods, significance (Kaggle T4 x2, 2026-11-14)

RGB-D, NYUv2 40 classes, 4000 iterations, seeds 0-2, final-checkpoint mIoU on the 654 test images.
Random is the Part 2 rerun in the same notebook (seed 0, 1, 2 = the committed random splits). It matches the
earlier baseline (15.02 vs 15.09 at 5%, 20.49 vs 20.77 at 10%), so the older runs are consistent with the new setup.
"Paired p" pairs runs that share a training seed (paired t-test, n = 3).

## 5% (40 images)

| method | mIoU (mean ± std) | per seed (0, 1, 2) | Δ vs random | paired p vs random | cover-rgbd minus this | paired p |
|---|---|---|---|---|---|---|
| random | 15.02 ± 0.80 | 15.89, 14.88, 14.30 | | | +2.87 | 0.05 |
| cdiv (C + diversity) | 14.98 ± 0.71 | 15.06, 15.65, 14.24 | −0.04 | 0.94 | +2.91 | 0.07 |
| typiclust (ICML 2022) | 15.63 ± 0.43 | 15.30, 15.47, 16.12 | +0.61 | 0.48 | +2.26 | **0.03** |
| kmeans-ctb (C tie-break) | 15.86 ± 0.09 | 15.81, 15.96, 15.81 | +0.84 | 0.22 | +2.03 | 0.06 |
| kmeans | 16.65 ± 0.63 | 17.04, 16.98, 15.92 | +1.62 | **0.03** | +1.24 | 0.23 |
| kmeans-rgbd | 16.67 ± 0.78 | 17.39, 16.79, 15.84 | +1.65 | **0.01** | +1.22 | 0.24 |
| cover-rgb (no depth) | 16.71 ± 0.76 | 17.25, 17.04, 15.85 | +1.69 | **0.02** | +1.18 | 0.27 |
| **cover-rgbd (ours)** | **17.89 ± 0.78** | 18.18, 17.01, 18.48 | **+2.87** | **0.05** | | |

## 10% (80 images)

| method | mIoU (mean ± std) | per seed (0, 1, 2) | Δ vs random | paired p vs random | cover-rgbd minus this | paired p |
|---|---|---|---|---|---|---|
| random | 20.49 ± 1.25 | 21.69, 20.58, 19.20 | | | +2.28 | **0.03** |
| kmeans | 20.90 ± 1.34 | 22.42, 20.35, 19.92 | +0.41 | 0.33 | +1.87 | 0.06 |
| typiclust | 21.09 ± 1.38 | 22.65, 20.59, 20.04 | +0.60 | 0.18 | +1.67 | 0.08 |
| kmeans-rgbd | 21.75 ± 0.79 | 21.12, 22.63, 21.49 | +1.26 | 0.30 | +1.02 | 0.24 |
| **cover-rgbd (ours)** | **22.77 ± 0.60** | 23.33, 22.83, 22.14 | **+2.28** | **0.03** | | |

## Pooled over both budgets (6 paired runs)

| cover-rgbd minus | per pair | mean | paired t p | Wilcoxon p |
|---|---|---|---|---|
| random | 2.29, 2.13, 4.18, 1.64, 2.25, 2.94 | +2.57 | 0.001 | 0.031 |
| typiclust | 2.88, 1.54, 2.36, 0.68, 2.24, 2.10 | +1.97 | 0.002 | 0.031 |
| kmeans | 1.14, 0.03, 2.56, 0.91, 2.48, 2.22 | +1.56 | 0.014 | 0.031 |
| kmeans-rgbd | 0.79, 0.22, 2.64, 2.21, 0.20, 0.65 | +1.12 | 0.047 | 0.031 |

cover-rgbd is ahead in all 6 pairs against every other method. With n = 6, 0.031 is the smallest p a Wilcoxon test can give.

## Findings

1. **cover-rgbd is the best method at both budgets.** It is +2.9 (5%) and +2.3 (10%) over random, and it is the only
   method that is clearly above random at 10%. Pooled over budgets it beats every baseline, kmeans included (p = 0.014).
   Within one budget, the lead over kmeans is still not significant with 3 seeds, so seeds 3-4 are needed.
2. **It beats the published cold-start baseline.** TypiClust is barely above random here (+0.6 at both budgets), and
   cover-rgbd beats it by +2.3 and +1.7 (pooled p = 0.002).
3. **Depth helps inside the coverage method.** cover-rgbd beats cover-rgb by +1.18 at 5%. That is consistent but not
   significant (p = 0.27, one seed pair tied). Adding depth to global features (kmeans-rgbd vs kmeans) changes nothing at
   5%. So depth seems to help when it describes regions, not whole images.
4. **C still adds nothing.** The C tie-break (kmeans-ctb, 15.86) is below plain kmeans (16.65), and cdiv equals random.
   This is the negative finding, alongside the check_c correlation of about 0.
5. **Rare-class pixels (labels used for analysis only).** At 5%, cover-rgbd puts the largest share of labelled pixels on the
   20 rarest classes: 13.61%, against 11.77% for random, 12.06% for kmeans and 11.98% for cover-rgb. kmeans-ctb is lowest
   (9.97%) and is also the weakest diversity method. At 10% cover-rgbd is still highest (13.42% vs 12.73% random), and at 25%
   the methods converge (12.1-12.6%). The class *presence* counts barely differ (38-39 of 40 at 5%), so the gain looks
   like it comes from **more pixels of rare classes, not more rare classes**. This is suggestive, not proven: across the
   7 methods at 5%, the rank correlation between rare-pixel share and mIoU is 0.5 (p = 0.25), and at 10% it is 0.
   Note: the stats table recomputes kmeans 5% on Kaggle, which may differ slightly from the Colab split that was trained.

## Next

* PART 3 (25%: random, kmeans, cover-rgbd; 9 runs, about 3.5 h). The notebook is set to PART = 3.
* Then seeds 3-4 for random, kmeans and cover-rgbd at 5% and 10% (12 runs). That is what can make the within-budget lead
  over kmeans significant.
