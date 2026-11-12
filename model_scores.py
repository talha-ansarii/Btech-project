"""Model-based scores, from models that are ALREADY trained (no extra training).

We reuse the baseline runs trained on the random 5% subset of a seed:
  runs/rand5_rgbd_s<seed>/final.pt   (RGB-D model)
  runs/rand5_rgb_s<seed>/final.pt    (RGB-only model, same 40 images)
and score every training image (one forward pass each, about 2 minutes on a T4):

  entropy(x) = mean over pixels of the RGB-D model's prediction entropy        (uncertainty baseline)
  mcomp(x)   = mean over pixels of JS( p_RGB-D || p_RGB )                       (model-based C:
               how much the prediction changes when depth is added; the per-pixel, dense version
               of the cross-modal disagreement in CCMA, 2026)
  feat_model = global-average-pooled decoder features of the RGB-D model (128-d), for coreset

Predictions are made at 240x320 (half resolution) to save time; scores are averages anyway.

  python model_scores.py --data data/nyuv2 --runs /content/drive/MyDrive/rgbx/runs --seed 0 \
                         --out cache/model_r5_s0.npz
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import NYUv2, load_meta
from model import RGBDFusionNet


def load_model(path, nc, mode, device):
    m = RGBDFusionNet(nc, mode=mode, pretrained=False)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.to(device).eval()


@torch.no_grad()
def score(data, rgbd_ckpt, rgb_ckpt, split="train", scale=0.5, batch=8, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    nc = load_meta(data)["num_classes"]
    m_d = load_model(rgbd_ckpt, nc, "rgbd", device)
    m_r = load_model(rgb_ckpt, nc, "rgb", device) if rgb_ckpt else None
    loader = DataLoader(NYUv2(data, split), batch_size=batch, num_workers=2)
    ent, js, feats = [], [], []
    for b in loader:
        rgb, depth = b["rgb"].to(device), b["depth"].to(device)
        if scale != 1:
            size = tuple(max(32, int(x * scale) // 32 * 32) for x in rgb.shape[-2:])
            rgb = F.interpolate(rgb, size, mode="bilinear", align_corners=False)
            depth = F.interpolate(depth, size, mode="nearest")
        with torch.autocast(device.type, enabled=amp):
            logit_d, f = m_d(rgb, depth, return_feats=True)
        logp_d = F.log_softmax(logit_d.float(), 1)
        p_d = logp_d.exp()
        ent.append(-(p_d * logp_d).sum(1).mean((1, 2)).cpu())
        feats.append(f.float().mean((2, 3)).cpu())
        if m_r is not None:
            with torch.autocast(device.type, enabled=amp):
                logit_r = m_r(rgb)
            logp_r = F.log_softmax(logit_r.float(), 1)
            p_r = logp_r.exp()
            logm = (0.5 * (p_d + p_r)).clamp_min(1e-8).log()
            jsd = 0.5 * (p_d * (logp_d - logm)).sum(1) + 0.5 * (p_r * (logp_r - logm)).sum(1)
            js.append(jsd.mean((1, 2)).cpu())
    out = {"entropy": torch.cat(ent).numpy(), "feat_model": torch.cat(feats).numpy()}
    if js:
        out["mcomp"] = torch.cat(js).numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs", required=True, help="folder holding rand5_rgbd_s<seed>/ and rand5_rgb_s<seed>/")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--round-name", default="rand5", help="prefix of the round-1 runs (default rand5)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    device = torch.device("cpu") if args.cpu else None
    ck = lambda mode: os.path.join(args.runs, f"{args.round_name}_{mode}_s{args.seed}", "final.pt")
    rgb_ck = ck("rgb") if os.path.exists(ck("rgb")) else None
    if rgb_ck is None:
        print(f"note: {ck('rgb')} not found, so mcomp (model-based C) is skipped")
    out = score(args.data, ck("rgbd"), rgb_ck, args.split, device=device)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, **out)
    msg = f"wrote {args.out}: entropy median {np.median(out['entropy']):.3f}"
    if "mcomp" in out:
        msg += f", mcomp median {np.median(out['mcomp']):.4f}"
    print(msg)


if __name__ == "__main__":
    main()
