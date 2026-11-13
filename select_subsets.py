"""Choose which training images to label, and write them as split files for train.py.

Two settings, both with the same budgets as the random baseline (5% = 40, 10% = 80 of 795):

COLD START (nothing labelled yet, so only label-free scores are possible). Budget k:
  ctop    the k images with the highest label-free complementarity C        (C only)
  kmeans  k-means into k clusters on ImageNet features, take the image
          nearest each cluster centre                                        (diversity only)
  cdiv    same k clusters, take the image with the highest C in each         (C + diversity)

  Diversity-led variants (added 2026-11-13 after C alone did not help; also cold start, no labels):
  typiclust    same k clusters, take the most "typical" image of each (highest local density, mean
               distance to its 20 nearest neighbours in the cluster; Hacohen et al. ICML 2022)  (baseline)
  kmeans-rgbd  k-means on RGB + depth-geometry features (features_rgbd.py: ImageNet ResNet-34 on the
               colour image and on surface normals from depth, concatenated), image nearest each centre
  cover-rgbd   region coverage: cluster the 6x8 region vectors (RGB + geometry, aligned) of the whole
               pool into --protos prototypes ("kinds of region"), then greedily add the image that most
               increases sum_p sqrt(#chosen regions of prototype p). The square root gives diminishing
               returns, so rare kinds of region (rare objects and surfaces) count as much as common ones:
               a label-free stand-in for class balance, which segmentation needs and image-level
               k-means does not give.
  cover-rgb    the same with RGB region vectors only                           (ablation: is depth needed?)
  kmeans-ctb   kmeans, but among the 3 images nearest each centre take the one with the highest C
               (ablation: C as a tie-break on top of diversity)

ROUND 2 (start from the random 5% of the same seed, add 40 images to reach 10%; the round-1 models
are the baseline runs rand5_rgbd_s<seed> and rand5_rgb_s<seed>, so no extra training is needed):
  r2-entropy   top-40 mean pixel entropy of the RGB-D model                  (uncertainty baseline)
  r2-coreset   k-center greedy on RGB-D decoder features (Sener & Savarese)  (diversity baseline)
  r2-mcomp     top-40 RGB vs RGB-D disagreement (model-based C)
  r2-mcdiv     40 clusters of the unlabelled pool, highest model-based C in each
  r2-cdiv      40 clusters of the unlabelled pool, highest label-free C in each
  r2-ctop, r2-kmeans   the label-free ablations, as above
  (random 10% already contains random 5%, so it IS the round-2 random baseline.)

Output: <out-dir>/<method>_frac<f>_seed<s>.json in the same format as splits/random_*.json,
so `train.py --subset ...` and summarize.py work unchanged.

  python select_subsets.py --cache cache --out-dir splits_sel --methods ctop kmeans cdiv --frac 0.05
  python select_subsets.py --cache cache --rgbd-feats cache/feats_rgbd.npz --out-dir splits_div \
                           --methods typiclust kmeans-rgbd cover-rgbd cover-rgb kmeans-ctb --frac 0.05
  python select_subsets.py --cache cache --out-dir splits_sel --round2 --from-dir splits \
                           --methods entropy coreset mcomp mcdiv cdiv --frac 0.10
"""
import argparse
import json
import os

import numpy as np

N_TRAIN = 795
LABEL_FREE = {"ctop", "kmeans", "cdiv", "typiclust", "kmeans-rgbd", "cover-rgbd", "cover-rgb", "kmeans-ctb"}
NEEDS_RGBD_FEATS = {"kmeans-rgbd", "cover-rgbd", "cover-rgb"}
MODEL_BASED = {"entropy", "coreset", "mcomp", "mcdiv"}


def kmeans_labels(x, k, seed):
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(x)
    return km.labels_, km.cluster_centers_


def per_cluster_best(x, score, k, seed):
    """k-means into k groups, keep the highest-scoring member of each group (positions into x)."""
    lab, _ = kmeans_labels(x, k, seed)
    return [int(np.flatnonzero(lab == c)[np.argmax(score[lab == c])]) for c in range(k) if (lab == c).any()]


def per_cluster_central(x, k, seed):
    lab, cen = kmeans_labels(x, k, seed)
    out = []
    for c in range(k):
        mem = np.flatnonzero(lab == c)
        if len(mem):
            out.append(int(mem[np.argmin(((x[mem] - cen[c]) ** 2).sum(1))]))
    return out


def per_cluster_typical(x, k, seed, knn=20):
    """TypiClust: in each k-means cluster, the member with the smallest mean distance to its nearest neighbours."""
    lab, _ = kmeans_labels(x, k, seed)
    out = []
    for c in range(k):
        mem = np.flatnonzero(lab == c)
        if len(mem) == 0:
            continue
        if len(mem) == 1:
            out.append(int(mem[0])); continue
        d = np.sqrt(((x[mem][:, None] - x[mem][None]) ** 2).sum(-1))
        nn = min(knn, len(mem) - 1)
        mean_nn = np.sort(d, 1)[:, 1:nn + 1].mean(1)          # column 0 is the image itself
        out.append(int(mem[np.argmin(mean_nn)]))
    return out


def per_cluster_central_tiebreak(x, score, k, seed, m=3):
    """k-means; among the m members nearest each centre, keep the one with the highest score."""
    lab, cen = kmeans_labels(x, k, seed)
    out = []
    for c in range(k):
        mem = np.flatnonzero(lab == c)
        if len(mem):
            near = mem[np.argsort(((x[mem] - cen[c]) ** 2).sum(1))[:m]]
            out.append(int(near[np.argmax(score[near])]))
    return out


def region_coverage(patches, k, seed, n_protos=256):
    """Greedy region coverage (positions into `patches`, which is N x G x D).

    Cluster all N*G region vectors into n_protos prototypes, count A[i, p] = regions of image i in prototype p,
    then add images one by one maximising F(S) = sum_p sqrt(sum_{i in S} A[i, p]) (monotone submodular,
    so greedy is within 1 - 1/e of the best set)."""
    from sklearn.cluster import MiniBatchKMeans
    n, g, dim = patches.shape
    flat = patches.reshape(-1, dim).astype(np.float32)
    km = MiniBatchKMeans(n_clusters=n_protos, n_init=3, random_state=seed, batch_size=4096).fit(flat)
    A = np.zeros((n, n_protos), np.float32)
    np.add.at(A, (np.repeat(np.arange(n), g), km.labels_), 1.0)
    cover, chosen, free = np.zeros(n_protos, np.float32), [], np.ones(n, bool)
    for _ in range(k):
        gain = (np.sqrt(cover[None] + A) - np.sqrt(cover[None])).sum(1)
        gain[~free] = -np.inf
        i = int(np.argmax(gain))
        chosen.append(i); free[i] = False; cover += A[i]
    print(f"    region coverage: {int((cover > 0).sum())}/{n_protos} prototypes covered by {k} images "
          f"(pool covers {int((A.sum(0) > 0).sum())})")
    return chosen


def kcenter_greedy(feat, labelled, pool, k):
    """Classic coreset selection: repeatedly add the pool image farthest from everything chosen so far."""
    f = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8)
    dist = np.full(len(pool), np.inf)
    for j in labelled:
        dist = np.minimum(dist, ((f[pool] - f[j]) ** 2).sum(1))
    chosen = []
    for _ in range(k):
        i = int(np.argmax(dist))
        chosen.append(pool[i])
        dist = np.minimum(dist, ((f[pool] - f[pool[i]]) ** 2).sum(1))
        dist[i] = -1
    return chosen


def pick(method, pool, k, seed, lf, ms=None, labelled=()):
    """Return k indices (into the 795 training images) chosen from `pool` by `method`."""
    pool = np.asarray(pool)
    C = lf["C"]
    if method == "ctop":
        return pool[np.argsort(-C[pool], kind="stable")[:k]].tolist()
    if method == "kmeans":
        return pool[per_cluster_central(lf["feat"][pool], k, seed)].tolist()
    if method == "cdiv":
        return pool[per_cluster_best(lf["feat"][pool], C[pool], k, seed)].tolist()
    if method == "typiclust":
        return pool[per_cluster_typical(lf["feat"][pool], k, seed)].tolist()
    if method == "kmeans-ctb":
        return pool[per_cluster_central_tiebreak(lf["feat"][pool], C[pool], k, seed)].tolist()
    if method in NEEDS_RGBD_FEATS:
        if "feat_rgb" not in lf:
            raise SystemExit(f"{method} needs --rgbd-feats (run features_rgbd.py first)")
        if method == "kmeans-rgbd":
            x = np.concatenate([lf["feat_rgb"], lf["feat_geo"]], 1) / np.sqrt(2)
            return pool[per_cluster_central(x[pool], k, seed)].tolist()
        p = lf["patch_rgb"] if method == "cover-rgb" else np.concatenate([lf["patch_rgb"], lf["patch_geo"]], 2)
        return pool[region_coverage(p[pool], k, seed, lf["n_protos"])].tolist()
    if ms is None:
        raise SystemExit(f"{method} needs model scores (round 2 only)")
    if method == "entropy":
        return pool[np.argsort(-ms["entropy"][pool], kind="stable")[:k]].tolist()
    if method == "mcomp":
        return pool[np.argsort(-ms["mcomp"][pool], kind="stable")[:k]].tolist()
    if method == "mcdiv":
        return pool[per_cluster_best(ms["feat_model"][pool], ms["mcomp"][pool], k, seed)].tolist()
    if method == "coreset":
        return [int(i) for i in kcenter_greedy(ms["feat_model"], list(labelled), pool, k)]
    raise SystemExit(f"unknown method {method}")


def load_labelfree(cache, c_variant, rgbd_feats=None, n_protos=256):
    z = np.load(os.path.join(cache, "labelfree_train.npz"))
    lf = {"C": z[c_variant], "feat": z["feat_imagenet"], "n_protos": n_protos}
    if rgbd_feats:
        lf.update(dict(np.load(rgbd_feats)))
    return lf


def write(out_dir, method, frac, seed, indices, extra):
    path = os.path.join(out_dir, f"{method}_frac{frac:.2f}_seed{seed}.json")
    with open(path, "w") as fh:
        json.dump(dict({"method": method, "fraction": frac, "seed": seed, "n": len(indices),
                        "indices": sorted(int(i) for i in indices)}, **extra), fh)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="folder with labelfree_train.npz and model_r5_s<seed>.npz")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--frac", type=float, required=True, help="final labelled fraction")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--round2", action="store_true", help="add to the random split given by --from-dir/--from-frac")
    ap.add_argument("--from-dir", default="splits")
    ap.add_argument("--from-frac", type=float, default=0.05)
    ap.add_argument("--c-variant", default="C_mean", choices=["C_mean", "C_ratio"])
    ap.add_argument("--rgbd-feats", help="feats_rgbd.npz from features_rgbd.py (for kmeans-rgbd, cover-*)")
    ap.add_argument("--protos", type=int, default=256, help="number of region prototypes for cover-*")
    ap.add_argument("--tag", default="", help="suffix for the method name, e.g. -ratio for ablations")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    lf = load_labelfree(args.cache, args.c_variant, args.rgbd_feats, args.protos)
    k_total = int(round(args.frac * N_TRAIN))

    for seed in args.seeds:
        labelled, ms, extra = [], None, {"c_variant": args.c_variant}
        if args.round2:
            src = os.path.join(args.from_dir, f"random_frac{args.from_frac:.2f}_seed{seed}.json")
            with open(src) as fh:
                labelled = json.load(fh)["indices"]
            ms_path = os.path.join(args.cache, f"model_r5_s{seed}.npz")
            ms = dict(np.load(ms_path)) if os.path.exists(ms_path) else None
            extra.update({"from": os.path.basename(src), "n_from": len(labelled)})
        pool = sorted(set(range(N_TRAIN)) - set(labelled))
        k = k_total - len(labelled)
        for method in args.methods:
            if not args.round2 and method in MODEL_BASED:
                raise SystemExit(f"{method} needs a trained model: use it with --round2")
            added = pick(method, pool, k, seed, lf, ms, labelled)
            assert len(set(added)) == k and not set(added) & set(labelled), method
            name = ("r2-" if args.round2 else "") + method + args.tag
            path = write(args.out_dir, name, args.frac, seed, labelled + added, dict(extra, added=sorted(added)))
            print(f"{path}: {len(labelled)} + {k}, mean C of added {lf['C'][added].mean():.4f} "
                  f"(pool {lf['C'][pool].mean():.4f})")


if __name__ == "__main__":
    main()
