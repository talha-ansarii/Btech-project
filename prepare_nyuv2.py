"""Turn the official NYU Depth V2 file into simple image folders.

Input : nyu_depth_v2_labeled.mat (official, ~2.8 GB, 1449 labelled frames)
        meta/splits.mat          (standard 795 train / 654 test split)
        meta/classMapping40.mat  (Gupta et al. mapping: 894 raw classes -> 40)
Output: <out>/train/{rgb,depth,label}/0000.png ...  (795 frames)
        <out>/test/{rgb,depth,label}/0000.png ...   (654 frames)
        <out>/meta.json  (class names, depth statistics, original frame ids)

Formats
  rgb   : 8-bit RGB png, 480x640
  depth : 16-bit png, depth in millimetres (the "filled" depth from the .mat)
  label : 8-bit png, values 0..39 for the 40 classes, 255 = unlabelled (ignored)

Usage
  python prepare_nyuv2.py --mat /path/nyu_depth_v2_labeled.mat --out data/nyuv2
  python prepare_nyuv2.py --fake --out data/fake_nyuv2   # tiny random dataset, to test the code
"""
import argparse
import json
import os

import numpy as np
import scipy.io as sio
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
IGNORE = 255


def class_names_and_map():
    m = sio.loadmat(os.path.join(HERE, "meta", "classMapping40.mat"))
    names = [str(x[0]) for x in m["className"][0]]
    # lut[raw] = class id in 0..39, raw 0 (unlabelled) -> 255
    lut = np.full(895, IGNORE, dtype=np.uint8)
    lut[1:] = m["mapClass"][0].astype(np.int64) - 1
    return names, lut


def standard_split():
    s = sio.loadmat(os.path.join(HERE, "meta", "splits.mat"))
    # 1-based frame numbers in the .mat file
    return s["trainNdxs"][:, 0].astype(int) - 1, s["testNdxs"][:, 0].astype(int) - 1


def save_frame(out_dir, i, rgb, depth_m, label40):
    Image.fromarray(rgb).save(os.path.join(out_dir, "rgb", f"{i:04d}.png"))
    depth_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
    Image.fromarray(depth_mm).save(os.path.join(out_dir, "depth", f"{i:04d}.png"))
    Image.fromarray(label40).save(os.path.join(out_dir, "label", f"{i:04d}.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", help="path to nyu_depth_v2_labeled.mat")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fake", action="store_true", help="write a tiny random dataset instead")
    args = ap.parse_args()

    names, lut = class_names_and_map()
    train_ids, test_ids = standard_split()
    for split in ("train", "test"):
        for sub in ("rgb", "depth", "label"):
            os.makedirs(os.path.join(args.out, split, sub), exist_ok=True)

    if args.fake:
        # Same file layout and counts as the real thing, but small random images,
        # so the whole pipeline can be checked on a laptop in a minute.
        rng = np.random.default_rng(0)
        n_train, n_test, h, w = len(train_ids), 20, 120, 160
        for split, n in (("train", n_train), ("test", n_test)):
            for i in range(n):
                # 4x4 grid of flat blocks; class k sits at depth 0.5 + 0.2k metres.
                # The label is only visible in depth (RGB is noise), so an RGB-D model
                # should learn it and an RGB-only model should not: a check that depth is used.
                cls = rng.integers(0, 40, (4, 4))
                label = np.kron(cls, np.ones((h // 4, w // 4))).astype(np.uint8)
                depth = (0.5 + 0.2 * label + rng.normal(0, 0.02, (h, w))).astype(np.float32)
                label[:5] = IGNORE
                rgb = (rng.integers(0, 255, (h, w, 3))).astype(np.uint8)
                save_frame(os.path.join(args.out, split), i, rgb, depth, label)
        train_ids, test_ids = train_ids[:n_train], test_ids[:n_test]
        depth_stats = {"mean": 4.75, "std": 2.45}
    else:
        import h5py  # the labelled .mat is a MATLAB v7.3 (HDF5) file

        f = h5py.File(args.mat, "r")
        images, depths, labels = f["images"], f["depths"], f["labels"]  # stored transposed
        dsum = dsq = dcount = 0.0
        for split, ids in (("train", train_ids), ("test", test_ids)):
            for i, k in enumerate(ids):
                rgb = np.transpose(images[k], (2, 1, 0)).astype(np.uint8)      # 480x640x3
                depth = np.transpose(depths[k], (1, 0)).astype(np.float32)     # metres
                raw = np.transpose(labels[k], (1, 0)).astype(np.int64)         # 0..894
                save_frame(os.path.join(args.out, split), i, rgb, depth, lut[raw])
                if split == "train":
                    dsum += depth.sum(); dsq += (depth ** 2).sum(); dcount += depth.size
                if i % 100 == 0:
                    print(f"{split}: {i}/{len(ids)}", flush=True)
        mean = dsum / dcount
        depth_stats = {"mean": float(mean), "std": float(np.sqrt(dsq / dcount - mean ** 2))}

    meta = {
        "num_classes": 40,
        "ignore_index": IGNORE,
        "class_names": names,
        "depth_mean_m": depth_stats["mean"],
        "depth_std_m": depth_stats["std"],
        "train_frame_ids": [int(x) + 1 for x in train_ids],  # 1-based ids in the .mat
        "test_frame_ids": [int(x) + 1 for x in test_ids],
        "fake": bool(args.fake),
    }
    with open(os.path.join(args.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"done: {len(train_ids)} train / {len(test_ids)} test written to {args.out}")


if __name__ == "__main__":
    main()
