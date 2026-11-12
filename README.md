# NYU Depth V2 baseline for data-efficient RGB-D segmentation

This folder is step 3 of the project plan: a standard RGB-D segmentation model, trained on
NYU Depth V2, first with all labels and then with only 5% and 10% of the labels picked at
random. These random-subset numbers are the "random selection" baseline that our
complementarity-aware selection has to beat later.

## What is in here

| file | what it does |
|---|---|
| `nyuv2_baseline_colab.ipynb` | **Start here.** Runs everything on a free Colab T4 GPU, saves to Google Drive. |
| `prepare_nyuv2.py` | Converts the official `nyu_depth_v2_labeled.mat` into png folders (795 train / 654 test, 40 classes). |
| `dataset.py` | Loads RGB + depth + label, with the usual augmentation (random scale 0.5 to 1.75, crop, flip). |
| `model.py` | The network (see below), in an RGB-D version and an RGB-only version. |
| `train.py` | Trains one run, evaluates on the 654 test images, logs everything. |
| `make_splits.py`, `splits/` | The fixed random 5% / 10% subsets for seeds 0, 1, 2 (already generated; do not regenerate). |
| `run_baseline.py` | Runs the whole plan, skipping runs that already finished (so Colab disconnects are harmless). |
| `summarize.py` | Makes the mean ± std results table. |
| `nyuv2_selection_colab.ipynb` and the files in "Step 4" below | Complementarity-aware selection (step 4). |
| `meta/` | `splits.mat` (standard train/test split) and `classMapping40.mat` (894 raw classes to 40), from the `nyuv2-meta-data` repository. |

## The model, in plain words

We use an **ESANet-style** network (Seichter et al., ICRA 2021, a published RGB-D model that was
designed to be efficient enough for a robot's onboard computer):

* Two ResNet-34 "encoders", one reads the colour image, one reads the depth map. Both start from
  ImageNet-pretrained weights, so they already know what edges and textures look like.
* After each of the 5 stages, the depth features are **added into** the colour stream. Before adding,
  each side passes through a small "squeeze-and-excitation" block, which learns how much to trust
  each feature channel. This lets the network lean on depth where colour is confusing (a white wall
  next to a white door) and ignore it where depth is noisy.
* A light decoder combines the coarse and fine features and outputs one of 40 classes per pixel.
* About 44M parameters; it trains on a single free T4 GPU in about an hour or two.

**Why this one and not the newest model (DFormer, CMX, ...)?** Our paper is about *which images to
label*, not about a better network. Reviewers want every selection method compared on the same,
well-known, reproducible model. ESANet-style is simple, fast on free GPUs, and easy to hook our
selection code into. The same code also has an **RGB-only** mode (one encoder, no depth), which we
need for two things: showing how much depth helps when labels are scarce, and computing the
"RGB vs RGB-D disagreement" part of our score later. Once the method works, we can repeat the key
numbers on a stronger model such as DFormer as a robustness check.

## The data, in plain words

* NYU Depth V2 has 1449 indoor photos with depth and pixel labels. Everyone uses the same split:
  **795 training images and 654 test images**, and the same **40-class** label set (Gupta et al. 2013).
* `prepare_nyuv2.py` reads the official file, applies the 894-to-40 class mapping, and marks
  unlabelled pixels as 255 so they are ignored in training and scoring.
* Depth is stored in millimetres in 16-bit png files and normalised to mean 0, std 1 when loaded.

## The experiment protocol (keep this fixed for every method we compare)

* **Runs:** full data (795 images) once; 5% (40 images) and 10% (80 images) random subsets with
  **3 seeds** each. The 10% set of a seed always contains its 5% set, like a real labelling round.
* **Training length** is counted in steps, not epochs: 10,000 steps for full data (about 100 epochs),
  4,000 steps for 5% and 10%. Batch 8, crop 480x640, SGD with momentum, learning rate 0.01,
  warm-up then "poly" decay, mixed precision.
* **Scoring:** mean IoU (mIoU) over the 40 classes on all 654 test images at full resolution, plus
  pixel accuracy and mean class accuracy. We report the **final** checkpoint. Picking the best
  checkpoint by looking at the test set would quietly inflate the numbers, so `train.py` keeps it
  only for reference.
* Seeds matter: with 40 images, two random subsets can differ by several mIoU points. That is why
  every low-label number is a mean ± std over 3 seeds, and why our method must beat random by more
  than that spread.

## What numbers to expect

These are rough expectations, not guarantees:

* **Full data, RGB-D:** roughly 43 to 48 mIoU. The ESANet paper reports about 50 with much longer
  training (500 epochs) and a heavier decoder. **Below about 40 means something is wrong**, tell me.
* **RGB-only full data:** a few points lower than RGB-D.
* **5% / 10%:** much lower, likely somewhere around 20 to 35 mIoU. The exact value is what we are measuring.

## How to run it

Open `nyuv2_baseline_colab.ipynb` in Google Colab (File, Upload notebook), choose a T4 GPU, and
run the cells top to bottom. It needs the code either on GitHub or uploaded to `MyDrive/rgbx/rgbx-baseline/`.

On your own machine or Kaggle the same steps are:

```bash
pip install -r requirements.txt
python prepare_nyuv2.py --mat nyu_depth_v2_labeled.mat --out data/nyuv2
python train.py --data data/nyuv2 --run smoke --iters 100 --eval-every 100 --max-test 20   # quick check
python run_baseline.py --data data/nyuv2 --out runs              # 7 RGB-D runs
python run_baseline.py --data data/nyuv2 --out runs --modes rgb  # 7 RGB-only runs (later)
python summarize.py
```

To test the code without a GPU or the real dataset:

```bash
python prepare_nyuv2.py --fake --out data/fake_nyuv2
python train.py --data data/fake_nyuv2 --run test --iters 300 --batch 4 --crop 112 144 --no-pretrained --cpu
```

On that fake dataset the class is only visible in depth, so the RGB-D model should learn it and the
`--mode rgb` model should stay near zero; this checks that depth really flows through the network.

## Step 4: complementarity-aware selection (new)

Notebook: **`nyuv2_selection_colab.ipynb`** (run it after the baseline notebook; it reuses the baseline's
data zip and trained weights on Drive).

### The score C, in plain words

An image is worth labelling for an RGB-D model when its **depth map shows object boundaries that the colour
image does not** (a white cupboard against a white wall, a dark chair in shadow). Those are the places where
depth can teach the network something RGB cannot. `complementarity.py` measures this without any labels or
trained model:

* depth edges `e_d`: gradient of **log** depth (so a step counts relative to distance), normalised by the
  95th percentile over the whole training pool and clipped to [0, 1];
* colour edges `e_r`: strongest RGB gradient, widened by a 5x5 max-filter so small RGB/depth misalignment
  does not count as "missing", normalised the same way;
* per pixel `m = e_d * (1 - e_r)` ("depth edge that RGB misses"); **C(x) = mean of m** over the image
  (`C_ratio`, the share of depth edges that RGB misses, is kept as an ablation).

It runs at 240x320 on the CPU in about a minute for all 795 images.

### Selection methods

| name | setting | how it picks | role in the paper |
|---|---|---|---|
| `random` | both | already done in the baseline | baseline |
| `cdiv` | cold start 5% | k-means into k groups on ImageNet features, highest C in each group | **ours** |
| `kmeans` | cold start 5% | same groups, image nearest the group centre | ablation: diversity only |
| `ctop` | cold start 5% | top-k by C | ablation: C only |
| `r2-cdiv` | round 2, 5% to 10% | random 5% of the seed + 40 by `cdiv` on the unlabelled rest | **ours** |
| `r2-entropy` | round 2 | + 40 with highest mean pixel entropy of the RGB-D model | uncertainty baseline |
| `r2-coreset` | round 2 | + 40 by k-center greedy on RGB-D features | diversity baseline (Sener & Savarese 2018) |
| `r2-mcomp` | round 2 | + 40 with highest JS divergence between RGB and RGB-D predictions | model-based C |
| `r2-mcdiv`, `r2-ctop`, `r2-kmeans` | round 2 | as above | ablations |

"Round 2" reuses the baseline models trained on the random 5% (`rand5_rgbd_s*`, `rand5_rgb_s*`), so the
model-based scores need **no extra training**, only one pass over the training images. Random 10% already
contains random 5%, so it is exactly the round-2 random baseline. The seed changes the k-means start and,
in round 2, the starting 5% and its models. `ctop` is the same set for every seed (only training noise varies).

### Files

| file | what it does |
|---|---|
| `complementarity.py` | label-free C (and ImageNet features) for every image → `cache/labelfree_train.npz` |
| `model_scores.py` | entropy, model-based C and features from the round-1 models → `cache/model_r5_s<seed>.npz` |
| `check_c.py` | 5-minute sanity check on the test set: does C rank images by how much depth helps? |
| `select_subsets.py` | writes `<method>_frac<f>_seed<s>.json` in the same format as `splits/` |
| `run_selection.py` | trains every (method, budget, seed), skipping finished runs, same settings as random |

Any split file works with `python train.py --subset <file>`, and `summarize.py` puts every method in one table.

### Extra cost

Scoring and split writing take about 10 minutes in total. The training runs are the real cost: priority 1 is
18 RGB-D runs (about 9 hours on a T4), priority 2 adds up to 18 optional ablation runs (12 RGB-D, 6 RGB-only).

Without a GPU or the real data, the whole pipeline can be checked on the fake dataset:

```bash
python prepare_nyuv2.py --fake --out data/fake
T="--iters 20 --batch 4 --crop 112 144 --no-pretrained --cpu --eval-every 20"
python train.py --data data/fake --out runs_t --run rand5_rgbd_s0 --subset splits/random_frac0.05_seed0.json $T
python train.py --data data/fake --out runs_t --run rand5_rgb_s0 --mode rgb --subset splits/random_frac0.05_seed0.json $T
python complementarity.py --data data/fake --out cache_t/labelfree_train.npz --no-pretrained
python model_scores.py --data data/fake --runs runs_t --seed 0 --out cache_t/model_r5_s0.npz --cpu
python select_subsets.py --cache cache_t --out-dir splits_t --methods ctop kmeans cdiv --frac 0.05 --seeds 0
python select_subsets.py --cache cache_t --out-dir splits_t --round2 --methods entropy coreset mcomp mcdiv cdiv --frac 0.10 --seeds 0
python run_selection.py --data data/fake --out runs_t --splits-dir splits_t --jobs cdiv:0.05 r2-mcomp:0.10 --seeds 0 -- $T
```

## Cite

* N. Silberman, D. Hoiem, P. Kohli, R. Fergus. Indoor Segmentation and Support Inference from RGBD Images. ECCV 2012 (the dataset).
* S. Gupta, P. Arbeláez, J. Malik. Perceptual Organization and Recognition of Indoor Scenes from RGB-D Images. CVPR 2013 (the 40 classes).
* O. Sener, S. Savarese. Active Learning for Convolutional Neural Networks: A Core-Set Approach. ICLR 2018 (coreset baseline).
* D. Seichter, M. Köhler, B. Lewandowski, T. Wengefeld, H.-M. Gross. Efficient RGB-D Semantic Segmentation for Indoor Scene Analysis. ICRA 2021 (ESANet).
