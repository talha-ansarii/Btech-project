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

| part | runs | contents | about |
|---|---|---|---|
| 1 | 15 | cover-rgbd 5% and 10%, kmeans-rgbd 5% and 10%, kmeans 10% | 4-5 h |
| 2 | 12 | typiclust 5% and 10%, cover-rgb 5%, kmeans-ctb 5% | 3.5-4 h |

27 runs in all, two at a time on the T4 x2. Both parts fit in one week of Kaggle GPU quota (30 h).
Already done and reused: random 5%/10%, kmeans 5%, cdiv 5%, and the round-2 baselines at 10%.

## What proves the claim

1. **Main claim** ("our method beats random and plain k-means at 5% and 10%"): cover-rgbd (or kmeans-rgbd)
   above random by more than the seed spread at both budgets, and at least level with kmeans.
   Needs part 1 only.
2. **Beats the published cold-start baseline**: ours ≥ typiclust at both budgets (part 2).
3. **Depth matters for selection**: cover-rgbd > cover-rgb, and kmeans-rgbd > kmeans (parts 1 + 2).
4. **C is not the answer** (negative finding): kmeans-ctb ≈ kmeans, cdiv ≈ random, plus the check_c correlation.
5. **Why it works**: the analyze_splits table shows the winning method covers more (rare) classes than random/kmeans.

## Decision after part 1

* If cover-rgbd or kmeans-rgbd clearly beats random at both budgets: run part 2, then add seeds 3-4 for
  random, kmeans and the winner at 5% (9 runs, about 2.5 h) so the main gap rests on 5 seeds.
* If neither beats plain kmeans: kmeans becomes the method and the paper leans on the analysis
  (class coverage, the 10% result, and the C finding). Part 2 is still worth running for the typiclust baseline.
