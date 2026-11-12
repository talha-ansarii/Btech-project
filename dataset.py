"""NYU Depth V2 dataset reader (works on the folders written by prepare_nyuv2.py)."""
import json
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # ImageNet statistics,
RGB_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)   # matching the pretrained encoder


def load_meta(root):
    with open(os.path.join(root, "meta.json")) as fh:
        return json.load(fh)


class NYUv2(Dataset):
    """Returns dict(rgb=3xHxW, depth=1xHxW, label=HxW long, index=int).

    indices: optional list of positions inside the split (0..794 for train) to keep.
             This is how the 5% / 10% labelled subsets are selected.
    train  : if True, apply random scale, crop and horizontal flip.
    """

    def __init__(self, root, split, indices=None, train=False, crop=(480, 640), scale=(0.5, 1.75)):
        self.dir = os.path.join(root, split)
        meta = load_meta(root)
        self.ignore = meta["ignore_index"]
        self.dmean, self.dstd = meta["depth_mean_m"], meta["depth_std_m"]
        n = len(os.listdir(os.path.join(self.dir, "rgb")))
        self.indices = list(range(n)) if indices is None else list(indices)
        self.train, self.crop, self.scale = train, crop, scale

    def __len__(self):
        return len(self.indices)

    def _load(self, i):
        name = f"{i:04d}.png"
        rgb = Image.open(os.path.join(self.dir, "rgb", name)).convert("RGB")
        depth = Image.open(os.path.join(self.dir, "depth", name))
        label = Image.open(os.path.join(self.dir, "label", name))
        return rgb, depth, label

    def _augment(self, rgb, depth, label):
        w, h = rgb.size
        s = random.uniform(*self.scale)
        nw, nh = int(w * s + 0.5), int(h * s + 0.5)
        rgb = rgb.resize((nw, nh), Image.BILINEAR)
        depth = depth.resize((nw, nh), Image.NEAREST)   # nearest keeps real depth values
        label = label.resize((nw, nh), Image.NEAREST)
        rgb, depth, label = np.asarray(rgb), np.asarray(depth), np.asarray(label)
        ch, cw = self.crop
        ph, pw = max(ch - nh, 0), max(cw - nw, 0)
        if ph or pw:  # pad if the scaled image is smaller than the crop
            rgb = np.pad(rgb, ((0, ph), (0, pw), (0, 0)))
            depth = np.pad(depth, ((0, ph), (0, pw)))
            label = np.pad(label, ((0, ph), (0, pw)), constant_values=self.ignore)
        y = random.randint(0, rgb.shape[0] - ch)
        x = random.randint(0, rgb.shape[1] - cw)
        rgb, depth, label = rgb[y:y + ch, x:x + cw], depth[y:y + ch, x:x + cw], label[y:y + ch, x:x + cw]
        if random.random() < 0.5:
            rgb, depth, label = rgb[:, ::-1], depth[:, ::-1], label[:, ::-1]
        return rgb, depth, label

    def __getitem__(self, k):
        i = self.indices[k]
        rgb, depth, label = self._load(i)
        if self.train:
            rgb, depth, label = self._augment(rgb, depth, label)
        else:
            rgb, depth, label = np.asarray(rgb), np.asarray(depth), np.asarray(label)
        rgb = (rgb.astype(np.float32) / 255.0 - RGB_MEAN) / RGB_STD
        d = depth.astype(np.float32) / 1000.0                     # millimetres -> metres
        d = (d - self.dmean) / self.dstd
        return {
            "rgb": torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))),
            "depth": torch.from_numpy(np.ascontiguousarray(d[None])),
            "label": torch.from_numpy(np.ascontiguousarray(label).astype(np.int64)),
            "index": i,
        }
