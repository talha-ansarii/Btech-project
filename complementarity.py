"""Label-free depth-edge complementarity score C (our main contribution), plus image features.

Idea in one sentence: an image is worth labelling for an RGB-D model when its depth map has object
boundaries that the colour image does not show (a white cupboard against a white wall, a dark chair
in shadow). Those are exactly the places where depth can teach the network something RGB cannot.

For every image (computed at half resolution, 240x320, which is enough for edges and fast):

  g_d(p) = |Sobel( log depth )|          depth edges, in log space so a 10 cm step counts the same
                                         near and far (relative depth discontinuity)
  g_r(p) = max over R,G,B of |Sobel(I)|  colour edges, then a 5x5 max-filter so that a depth edge
                                         a couple of pixels away from its colour edge still counts
                                         as "visible in RGB" (depth and RGB are not perfectly aligned)
  e_d = clip(g_d / P95(g_d), 0, 1)       normalised with ONE threshold for the whole pool (95th
  e_r = clip(g_r / P95(g_r), 0, 1)       percentile), so images with more strong edges score higher

  m(p) = e_d(p) * (1 - e_r(p))           "depth edge that RGB misses", per pixel, in [0, 1]

  C_mean(x)  = mean_p m(p)                         amount of depth-only structure   (default C)
  C_ratio(x) = sum_p m(p) / sum_p e_d(p)           share of the depth edges that RGB misses

No labels and no trained model are needed, so C works at the very first labelling round (cold start).
A 16-pixel frame is ignored because NYUv2 images have a white border.

The script also stores ImageNet ResNet-34 features of the RGB images (global average pooled, 512-d),
which the selection step uses to keep the chosen images diverse.

  python complementarity.py --data data/nyuv2 --split train --out cache/labelfree_train.npz
"""
import argparse
import os

import numpy as np
from PIL import Image
from scipy import ndimage

DEPTH_MIN_M, DEPTH_MAX_M = 0.3, 10.0


def _resize(arr, size, mode):
    return np.asarray(Image.fromarray(arr).resize(size, mode))


def edge_maps(rgb_u8, depth_mm, scale=0.5, border=16, sigma=1.0, dilate=5):
    """Raw (un-normalised) depth-edge and colour-edge strength maps for one image."""
    h, w = rgb_u8.shape[:2]
    size = (int(w * scale), int(h * scale))
    rgb = _resize(rgb_u8, size, Image.BILINEAR).astype(np.float32) / 255.0
    depth = _resize(depth_mm.astype(np.float32), size, Image.BILINEAR) / 1000.0
    logd = np.log(np.clip(depth, DEPTH_MIN_M, DEPTH_MAX_M))

    def grad(x):
        x = ndimage.gaussian_filter(x, sigma)
        return np.hypot(ndimage.sobel(x, 0), ndimage.sobel(x, 1)) / 8.0   # /8: Sobel -> per-pixel step

    g_d = grad(logd)
    g_r = np.max(np.stack([grad(rgb[..., c]) for c in range(3)]), 0)
    g_r = ndimage.maximum_filter(g_r, size=dilate)
    b = int(round(border * scale))
    return g_d[b:-b, b:-b], g_r[b:-b, b:-b]


def complementarity_map(g_d, g_r, p_d, p_r):
    e_d = np.clip(g_d / p_d, 0, 1)
    e_r = np.clip(g_r / p_r, 0, 1)
    return e_d * (1 - e_r), e_d, e_r


def load_pair(split_dir, i):
    name = f"{i:04d}.png"
    rgb = np.asarray(Image.open(os.path.join(split_dir, "rgb", name)).convert("RGB"))
    depth = np.asarray(Image.open(os.path.join(split_dir, "depth", name)))
    return rgb, depth


def labelfree_scores(data, split="train", pct=95.0, stats=None, verbose=True):
    """Returns dict with C_mean, C_ratio, depth_edge, rgb_edge (one value per image) and the
    percentile thresholds used (so the same thresholds can be reused on another split)."""
    split_dir = os.path.join(data, split)
    n = len(os.listdir(os.path.join(split_dir, "rgb")))
    maps = []
    for i in range(n):
        maps.append(tuple(m.astype(np.float16) for m in edge_maps(*load_pair(split_dir, i))))
        if verbose and (i + 1) % 200 == 0:
            print(f"  edges {i + 1}/{n}", flush=True)
    if stats is None:
        rng = np.random.default_rng(0)
        sample = lambda k: np.concatenate([rng.choice(m[k].ravel(), 4000).astype(np.float32) for m in maps])
        stats = {"p_d": float(np.percentile(sample(0), pct)), "p_r": float(np.percentile(sample(1), pct))}
    out = {k: np.zeros(n, np.float32) for k in ("C_mean", "C_ratio", "depth_edge", "rgb_edge")}
    for i, (g_d, g_r) in enumerate(maps):
        m, e_d, e_r = complementarity_map(g_d.astype(np.float32), g_r.astype(np.float32), stats["p_d"], stats["p_r"])
        out["C_mean"][i] = m.mean()
        out["C_ratio"][i] = m.sum() / (e_d.sum() + 1e-6)
        out["depth_edge"][i] = e_d.mean()
        out["rgb_edge"][i] = e_r.mean()
    out.update(p_d=stats["p_d"], p_r=stats["p_r"])
    return out


def imagenet_features(data, split="train", pretrained=True, batch=16, device=None):
    """512-d ImageNet ResNet-34 embedding of each RGB image (label-free, used for diversity)."""
    import timm
    import torch
    from torch.utils.data import DataLoader
    from dataset import NYUv2

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = timm.create_model("resnet34", pretrained=pretrained, num_classes=0).to(device).eval()
    loader = DataLoader(NYUv2(data, split), batch_size=batch, num_workers=2)
    feats = []
    with torch.no_grad(), torch.autocast(device.type, enabled=device.type == "cuda"):
        for b in loader:
            feats.append(net(b["rgb"].to(device)).float().cpu())
    f = torch.cat(feats).numpy()
    return f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True, help=".npz file to write")
    ap.add_argument("--stats-from", help="reuse the percentile thresholds stored in this .npz")
    ap.add_argument("--no-features", action="store_true", help="skip the ImageNet features")
    ap.add_argument("--no-pretrained", action="store_true", help="random-init features (testing only)")
    args = ap.parse_args()

    stats = None
    if args.stats_from:
        z = np.load(args.stats_from)
        stats = {"p_d": float(z["p_d"]), "p_r": float(z["p_r"])}
    print(f"label-free complementarity on {args.split} ...", flush=True)
    out = labelfree_scores(args.data, args.split, stats=stats)
    if not args.no_features:
        print("ImageNet features ...", flush=True)
        out["feat_imagenet"] = imagenet_features(args.data, args.split, pretrained=not args.no_pretrained)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, **out)
    c = out["C_mean"]
    print(f"wrote {args.out}: {len(c)} images, C_mean min {c.min():.4f} median {np.median(c):.4f} max {c.max():.4f}; "
          f"thresholds p_d {out['p_d']:.4f} p_r {out['p_r']:.4f}")


if __name__ == "__main__":
    main()
