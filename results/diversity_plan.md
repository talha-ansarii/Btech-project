# Diversity-led selection: method, runs, and what each run proves

Pivot (2026-11-13): the label-free score C did not predict where depth helps (check_c: Spearman about 0),
and C + diversity (cdiv) did not beat random. Plain k-means on ImageNet features was the only method that beat
random at 5% (16.65 ± 0.63 vs 15.09 ± 0.99). The paper is now about **label-free, diversity-led selection for
RGB-D segmentation**, with the C result kept as a short negative finding.

## Methods (all cold start: no labels, no trained model)

| name | what it picks | role in the paper |
|---|---|---|
| random | random images (existing splits) | baseline |
| kmeans | k-means on ImageNet RGB features, image nearest each centre | existing; diversity baseline |
| typiclust | same clusters, most typical (densest) image in each (Hacohen et al., ICML 2022) | published cold-start baseline |
| kmeans-rgbd | k-means on RGB + depth-geometry features (ResNet-34 on colour and on surface normals) | ours, simple version |
| cover-rgbd | greedy coverage of region prototypes (6x8 region vectors, RGB + geometry), sqrt diminishing returns | ours, main candidate |
| cover-rgb | cover-rgbd without the geometry features | ablation: does depth help selection? |
| kmeans-ctb | kmeans, C as a tie-break among the 3 images nearest each centre | ablation: does C add anything? |

Why region coverage: segmentation is scored per class, so a labelled set should contain many kinds of
objects and surfaces, including rare ones. Image-level k-means follows where images are dense; counting
regions with a square root makes the 10th "wall" region worth far less than the first "whiteboard" region.
`analyze_splits.py` checks this with the ground-truth labels (analysis only): classes present, rare classes
present, class entropy of each chosen set.

## Runs (RGB-D, 4000 iters, seeds 0-2, Kaggle notebook `kaggle/diversity_kaggle.ipynb`)

| part | runs | contents | about | status |
|---|---|---|---|---|
| 1 | 15 | cover-rgbd 5% and 10%, kmeans-rgbd 5% and 10%, kmeans 10% | 5 h | done 2026-11-14, see diversity_part1/ |
| 2 | 18 | random rerun 5% and 10%, typiclust 5% and 10%, cover-rgb 5%, kmeans-ctb 5% | 5.6 h | done 2026-11-14, see diversity_part2/ |
| 3 | 9 | 25% budget (199 images): random, kmeans, cover-rgbd | 3.5 h | next |

Two runs train at a time on the T4 x2, and each takes 36 to 40 minutes that way (not 26: the two runs share the CPUs).
Parts 1 to 3 total about 15 GPU hours of Kaggle's 30 h weekly quota.
The random 25% splits are nested in the existing 5% and 10% ones (make_splits.py, same permutation).

## What proves the claim

1. **Main claim** ("our method beats random and plain k-means at 5% and 10%"): cover-rgbd (or kmeans-rgbd)
   above random by more than the seed spread at both budgets, and at least level with kmeans.
   Needs part 1 only.
2. **Beats the published cold-start baseline**: ours ≥ typiclust at both budgets (part 2).
3. **Depth matters for selection**: cover-rgbd > cover-rgb, and kmeans-rgbd > kmeans (parts 1 + 2).
4. **C is not the answer** (negative finding): kmeans-ctb ≈ kmeans, cdiv ≈ random, plus the check_c correlation.
5. **Why it works**: the analyze_splits table shows the winning method covers more (rare) classes than random/kmeans.

## Decision after part 1

* Outcome: cover-rgbd beats random at both budgets (+2.8 at 5%, +2.0 at 10%, p < 0.05), but its lead over kmeans
  (+1.2, +1.9) is not significant with 3 seeds. So after parts 2 and 3, add seeds 3-4 for random, kmeans and
  cover-rgbd at 5% and 10% (12 runs, about 4 h).
* If neither beats plain kmeans: kmeans becomes the method and the paper leans on the analysis
  (class coverage, the 10% result, and the C finding). Part 2 is still worth running for the typiclust baseline.
