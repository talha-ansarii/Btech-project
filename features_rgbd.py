"""Label-free RGB and depth-geometry embeddings of every training image, for diversity-led selection.

For each image we run the same ImageNet ResNet-34 that initialises the segmentation encoders on two inputs:
  RGB       the colour image (ImageNet normalisation)
  geometry  surface normals computed from the depth map, shown to the network as a 3-channel image.
            Normals are a standard way to give an RGB-pretrained network depth (as in HHA, Gupta et al.
            ECCV 2014): they encode the layout (floor, walls, table tops) and are independent of distance.

and keep two things from the last stage (layer4, 512 channels):
  feat_rgb, feat_geo      global average pooled, L2-normalised, 512-d               (one vector per image)
  patch_rgb, patch_geo    average pooled to a 6x8 grid, L2-normalised per cell,   (48 region vectors per image,
                          stored as float16, N x 48 x 512                           aligned between the two)

The global vectors are for image-level k-means (kmeans-rgbd); the region vectors are for region coverage
(cover-rgbd), which counts how many distinct kinds of region (object/surface types) the chosen set contains.
No labels and no trained model are used. About 2 minutes on a T4.

  python features_rgbd.py --data data/nyuv2 --out cache/feats_rgbd.npz
"""
import argparse
import os

import numpy as np
from PIL import Image
from scipy import ndimage

RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
RGB_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# Kinect intrinsics of NYU Depth V2 for 640x480 frames (Silberman et al. 2012 toolbox); scaled for other sizes
FX, FY, CX, CY, W0, H0 = 518.86, 519.47, 325.58, 253.74, 640, 480
GRID = (6, 8)


def surface_normals(depth_mm, sigma=2.0):
    """Per-pixel unit normals (H x W x 3, camera frame) from a depth map in millimetres."""
    h, w = depth_mm.shape
    z = np.clip(depth_mm.astype(np.float32) / 1000.0, 0.3, 10.0)
    z = ndimage.gaussian_filter(z, sigma)
    fx, fy, cx, cy = FX * w / W0, FY * h / H0, CX * w / W0, CY * h / H0
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    x, y = (u - cx) * z / fx, (v - cy) * z / fy
    p = np.stack([x, y, z], -1)
    du, dv = np.gradient(p, axis=1), np.gradient(p, axis=0)
    n = np.cross(du, dv)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-8
    n[n[..., 2] > 0] *= -1          # make every normal face the camera
    return n


def load(split_dir, i):
    name = f"{i:04d}.png"
    rgb = np.asarray(Image.open(os.path.join(split_dir, "rgb", name)).convert("RGB"), dtype=np.float32) / 255.0
    depth = np.asarray(Image.open(os.path.join(split_dir, "depth", name)))
    geo = (surface_normals(depth) + 1.0) / 2.0          # [-1, 1] -> [0, 1], like an image
    norm = lambda a: ((a - RGB_MEAN) / RGB_STD).transpose(2, 0, 1)
    return norm(rgb).astype(np.float32), norm(geo.astype(np.float32)).astype(np.float32)


def embed(data, split="train", pretrained=True, batch=16, device=None, verbose=True):
    import timm
    import torch
    import torch.nn.functional as F

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = timm.create_model("resnet34", pretrained=pretrained, features_only=True, out_indices=(4,)).to(device).eval()
    split_dir = os.path.join(data, split)
    n = len(os.listdir(os.path.join(split_dir, "rgb")))
    out = {k: [] for k in ("feat_rgb", "feat_geo", "patch_rgb", "patch_geo")}
    l2 = lambda t: t / (t.norm(dim=-1, keepdim=True) + 1e-8)
    for s in range(0, n, batch):
        pairs = [load(split_dir, i) for i in range(s, min(n, s + batch))]
        for k, idx in (("rgb", 0), ("geo", 1)):
            x = torch.from_numpy(np.stack([p[idx] for p in pairs])).to(device)
            with torch.no_grad(), torch.autocast(device.type, enabled=device.type == "cuda"):
                f = net(x)[0].float()                                  # B x 512 x h x w
            out[f"feat_{k}"].append(l2(f.mean((2, 3))).cpu().numpy())
            g = F.adaptive_avg_pool2d(f, GRID).flatten(2).transpose(1, 2)   # B x 48 x 512
            out[f"patch_{k}"].append(l2(g).cpu().numpy().astype(np.float16))
        if verbose and (s // batch) % 10 == 0:
            print(f"  features {min(n, s + batch)}/{n}", flush=True)
    return {k: np.concatenate(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-pretrained", action="store_true", help="random-init network (testing only)")
    args = ap.parse_args()
    out = embed(args.data, args.split, pretrained=not args.no_pretrained)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, **out)
    sim = (out["feat_rgb"] * out["feat_geo"]).sum(1)
    print(f"wrote {args.out}: {len(sim)} images, patches {out['patch_rgb'].shape}; "
          f"mean cos(rgb, geo) of the same image {sim.mean():.3f}")


if __name__ == "__main__":
    main()
