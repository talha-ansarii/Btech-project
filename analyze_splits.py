"""Describe what each selected subset contains, using the ground-truth labels (for ANALYSIS only:
no selection method ever sees these labels).

Per split file: how many of the 40 classes appear at all, how many of the 20 rarest classes (rarest by
pixel count over the full 795-image training set) appear, and the normalised entropy of the class pixel
distribution (1 = perfectly balanced over the classes present in the full set). Printed as a table of
mean over seeds per (method, fraction). This is the evidence for "diversity / region coverage buys
class coverage", next to the mIoU table.

  python analyze_splits.py --data data/nyuv2 --splits splits splits_sel splits_div --md split_stats.md
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
from PIL import Image


def class_histograms(data, nc, ignore):
    lab_dir = os.path.join(data, "train", "label")
    n = len(os.listdir(lab_dir))
    H = np.zeros((n, nc), np.int64)
    for i in range(n):
        y = np.asarray(Image.open(os.path.join(lab_dir, f"{i:04d}.png"))).ravel()
        y = y[(y != ignore) & (y < nc)]
        H[i] = np.bincount(y, minlength=nc)[:nc]
    return H


def stats(H, idx, rare):
    h = H[idx].sum(0).astype(np.float64)
    present = h > 0
    p = h[present] / h.sum()
    ent = float(-(p * np.log(p)).sum() / np.log((H.sum(0) > 0).sum()))
    return int(present.sum()), int(present[rare].sum()), ent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--splits", nargs="+", required=True, help="folders with *_frac*_seed*.json")
    ap.add_argument("--md")
    args = ap.parse_args()
    with open(os.path.join(args.data, "meta.json")) as fh:
        meta = json.load(fh)
    H = class_histograms(args.data, meta["num_classes"], meta["ignore_index"])
    rare = np.argsort(H.sum(0))[:20]
    groups = defaultdict(list)
    for d in args.splits:
        for f in sorted(glob.glob(os.path.join(d, "*_frac*_seed*.json"))):
            m = re.match(r"(.+)_frac([\d.]+)_seed(\d+)\.json", os.path.basename(f))
            with open(f) as fh:
                idx = json.load(fh)["indices"]
            groups[(float(m[2]), m[1])].append(stats(H, idx, rare))
    lines = ["| labels | method | seeds | classes present (of 40) | rare classes present (of 20) | class entropy |",
             "|---|---|---|---|---|---|"]
    for (frac, method), v in sorted(groups.items()):
        a = np.array(v, np.float64)
        lines.append(f"| {frac * 100:.0f}% | {method} | {len(v)} | {a[:, 0].mean():.1f} | {a[:, 1].mean():.1f} | "
                     f"{a[:, 2].mean():.3f} |")
    print("\n".join(lines))
    if args.md:
        with open(args.md, "w") as fh:
            fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
