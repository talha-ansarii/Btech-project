"""Cheap sanity check (about 5 minutes, no training): does C point at images where depth really helps?

On the 654 TEST images (never used for selection) we compare two models that are already trained,
full_rgbd_s0 and full_rgb_s0. For each test image:
  gain(x) = per-image mIoU of the RGB-D model  -  per-image mIoU of the RGB-only model
and we ask whether the label-free score C (computed with the training-set thresholds) ranks the
images by gain. Output: Spearman correlation of gain with C and with simpler scores
(depth-edge density, colour-edge density), and the mean gain in the top vs bottom quarter by C.

If C correlates with gain clearly better than plain depth-edge density does, that is early evidence
for the paper's main claim, before spending GPU hours on the selection runs.

  python check_c.py --data data/nyuv2 --runs $RUNS --cache cache
"""
import argparse
import json
import os

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from complementarity import labelfree_scores
from dataset import NYUv2, load_meta
from model_scores import load_model


@torch.no_grad()
def per_image_miou(model, loader, nc, mode, device):
    amp = device.type == "cuda"
    out = []
    for b in loader:
        rgb, depth, label = b["rgb"].to(device), b["depth"].to(device), b["label"].to(device)
        with torch.autocast(device.type, enabled=amp):
            pred = model(rgb, depth if mode == "rgbd" else None).argmax(1)
        for p, l in zip(pred, label):
            keep = l < nc
            conf = torch.bincount(l[keep] * nc + p[keep], minlength=nc * nc).view(nc, nc).double()
            tp = conf.diag()
            union = conf.sum(0) + conf.sum(1) - tp
            present = conf.sum(1) > 0                        # classes in the ground truth of this image
            out.append(float((tp[present] / union[present].clamp(min=1)).mean() * 100) if present.any() else np.nan)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--cache", required=True, help="folder with labelfree_train.npz (for the thresholds)")
    ap.add_argument("--rgbd-run", default="full_rgbd_s0")
    ap.add_argument("--rgb-run", default="full_rgb_s0")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    nc = load_meta(args.data)["num_classes"]

    z = np.load(os.path.join(args.cache, "labelfree_train.npz"))
    lf = labelfree_scores(args.data, "test", stats={"p_d": float(z["p_d"]), "p_r": float(z["p_r"])})
    loader = DataLoader(NYUv2(args.data, "test"), batch_size=4, num_workers=2)
    miou = {}
    for mode, run in (("rgbd", args.rgbd_run), ("rgb", args.rgb_run)):
        m = load_model(os.path.join(args.runs, run, "final.pt"), nc, mode, device)
        miou[mode] = per_image_miou(m, loader, nc, mode, device)
        del m
    gain = miou["rgbd"] - miou["rgb"]
    ok = np.isfinite(gain)
    res = {"n_images": int(ok.sum()), "mean_gain": float(gain[ok].mean())}
    print(f"{ok.sum()} test images, mean per-image gain from depth {gain[ok].mean():+.2f} mIoU")
    for key in ("C_mean", "C_ratio", "depth_edge", "rgb_edge"):
        rho, p = spearmanr(lf[key][ok], gain[ok])
        res[f"spearman_{key}"] = [float(rho), float(p)]
        print(f"  Spearman(gain, {key:10s}) = {rho:+.3f}   (p = {p:.1e})")
    c, g = lf["C_mean"][ok], gain[ok]
    q1, q3 = np.quantile(c, [0.25, 0.75])
    res["gain_top_quarter_C"], res["gain_bottom_quarter_C"] = float(g[c >= q3].mean()), float(g[c <= q1].mean())
    print(f"  mean gain, top quarter by C: {g[c >= q3].mean():+.2f}   bottom quarter: {g[c <= q1].mean():+.2f}")
    np.savez(os.path.join(args.cache, "check_c_test.npz"), gain=gain, miou_rgbd=miou["rgbd"], miou_rgb=miou["rgb"],
             **{k: lf[k] for k in ("C_mean", "C_ratio", "depth_edge", "rgb_edge")})
    with open(os.path.join(args.cache, "check_c.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
