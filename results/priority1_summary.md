# Priority 1 results (RGB-D, NYUv2 40 classes, 4000 iters, mean ± std over seeds 0-2, final mIoU)

| setting | method | mIoU | per seed (0, 1, 2) |
|---|---|---|---|
| 5% (40 img) | random (baseline) | 15.09 ± 0.99 | |
| 5% | cdiv (ours) | 14.98 ± 0.71 | 15.06, 15.65, 14.24 |
| 5% | kmeans (diversity only) | 16.65 ± 0.63 | 17.04, 16.98, 15.92 |
| 10% (80 img) | random (baseline) | 20.77 ± 0.54 | |
| 10% | r2-cdiv (ours) | 20.38 ± 0.46 | 20.86, 19.94, 20.33 |
| 10% | r2-entropy | 20.82 ± 1.28 | 22.13, 19.57, 20.77 |
| 10% | r2-coreset | 20.39 ± 0.87 | 20.29, 21.31, 19.57 |
| 10% | r2-mcomp | 20.83 ± 0.90 | 20.26, 20.35, 21.87 |

Sources: results/colab_partial (8 runs), results/kaggle_part1, results/kaggle_part2. Random baselines from the earlier baseline runs.
