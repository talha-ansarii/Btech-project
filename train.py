"""Train and evaluate one run.

Examples
  # full training set (795 images), RGB-D
  python train.py --data data/nyuv2 --run full_rgbd_s0 --seed 0 --iters 10000

  # 5% random labelled subset, seed 1
  python train.py --data data/nyuv2 --run rand5_rgbd_s1 --seed 1 --iters 4000 \
                  --subset splits/random_frac0.05_seed1.json

  # same, RGB only (no depth)
  python train.py ... --mode rgb

Every run writes runs/<run>/ with: config.json, log.csv (loss curve), metrics.json
(final mIoU, pixel accuracy, per-class IoU), best.pt (weights). It also appends one line
to runs/results.csv so all runs can be compared with summarize.py.
Training is counted in iterations (not epochs) so that runs with 40 or 795 images get a
comparable amount of compute.
"""
import argparse
import csv
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import NYUv2, load_meta
from model import RGBDFusionNet


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def endless(loader):
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def evaluate(model, loader, num_classes, device, mode, amp):
    """Standard NYUv2 metrics from a confusion matrix over all test pixels (unlabelled ignored)."""
    model.eval()
    conf = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)
    for b in loader:
        rgb, depth, label = b["rgb"].to(device), b["depth"].to(device), b["label"].to(device)
        with torch.autocast(device.type, enabled=amp):
            pred = model(rgb, depth if mode == "rgbd" else None).argmax(1)
        keep = label < num_classes
        conf += torch.bincount(label[keep] * num_classes + pred[keep],
                               minlength=num_classes ** 2).view(num_classes, num_classes)
    model.train()
    conf = conf.double()
    tp = conf.diag()
    iou = tp / (conf.sum(0) + conf.sum(1) - tp).clamp(min=1)
    acc_c = tp / conf.sum(1).clamp(min=1)
    present = conf.sum(1) > 0
    return {
        "mIoU": float(iou[present].mean() * 100),
        "pixel_acc": float(tp.sum() / conf.sum().clamp(min=1) * 100),
        "mean_acc": float(acc_c[present].mean() * 100),
        "per_class_iou": [round(float(x) * 100, 2) for x in iou],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", required=True, help="name of this run (folder under --out)")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--mode", default="rgbd", choices=["rgbd", "rgb"])
    ap.add_argument("--subset", help="json file with labelled train indices (default: all 795)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--crop", type=int, nargs=2, default=[480, 640])
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-pretrained", action="store_true", help="random init (only for testing)")
    ap.add_argument("--max-test", type=int, default=0, help="evaluate on first N test images only (testing)")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    amp = device.type == "cuda"
    torch.backends.cudnn.benchmark = True
    meta = load_meta(args.data)
    nc = meta["num_classes"]

    indices, subset_name = None, "full"
    if args.subset:
        with open(args.subset) as fh:
            indices = json.load(fh)["indices"]
        subset_name = os.path.splitext(os.path.basename(args.subset))[0]

    train_ds = NYUv2(args.data, "train", indices=indices, train=True, crop=tuple(args.crop))
    test_ds = NYUv2(args.data, "test")
    if args.max_test:
        test_ds.indices = test_ds.indices[:args.max_test]
    g = torch.Generator(); g.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=len(train_ds) >= args.batch,
                              num_workers=args.workers, pin_memory=amp, generator=g, persistent_workers=args.workers > 0)
    test_loader = DataLoader(test_ds, batch_size=4, num_workers=args.workers, pin_memory=amp)

    model = RGBDFusionNet(nc, mode=args.mode, pretrained=not args.no_pretrained).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    # linear warm-up, then "poly" decay (standard for segmentation)
    lr_at = lambda it: (it + 1) / args.warmup if it < args.warmup else (1 - (it - args.warmup) / max(1, args.iters - args.warmup)) ** 0.9
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    scaler = torch.amp.GradScaler(enabled=amp)

    run_dir = os.path.join(args.out, args.run)
    os.makedirs(run_dir, exist_ok=True)
    config = dict(vars(args), n_labelled=len(train_ds), subset_name=subset_name, device=str(device))
    with open(os.path.join(run_dir, "config.json"), "w") as fh:
        json.dump(config, fh, indent=1)
    print(f"run {args.run}: {len(train_ds)} labelled images, {len(test_ds)} test, {args.mode}, {device}")

    log = open(os.path.join(run_dir, "log.csv"), "w", newline="")
    logw = csv.writer(log); logw.writerow(["iter", "loss", "lr", "mIoU", "sec"])
    best, t0, running = None, time.time(), 0.0
    batches = endless(train_loader)
    model.train()
    for it in range(args.iters):
        b = next(batches)
        rgb, depth, label = b["rgb"].to(device, non_blocking=True), b["depth"].to(device, non_blocking=True), b["label"].to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=amp):
            logits = model(rgb, depth if args.mode == "rgbd" else None)
            loss = F.cross_entropy(logits.float(), label, ignore_index=meta["ignore_index"])
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update(); sched.step()
        if not math.isfinite(loss.item()):
            raise RuntimeError(f"loss became {loss.item()} at iter {it}; try a lower --lr")
        running = 0.98 * running + 0.02 * loss.item() if it else loss.item()
        last = it + 1 == args.iters
        if (it + 1) % 50 == 0 or last:
            print(f"iter {it + 1}/{args.iters} loss {running:.3f} lr {opt.param_groups[0]['lr']:.5f} "
                  f"{(time.time() - t0) / (it + 1):.2f}s/it", flush=True)
        miou = ""
        if (it + 1) % args.eval_every == 0 or last:
            m = evaluate(model, test_loader, nc, device, args.mode, amp)
            miou = round(m["mIoU"], 2)
            print(f"  eval @ {it + 1}: mIoU {m['mIoU']:.2f}  pixel acc {m['pixel_acc']:.2f}", flush=True)
            if best is None or m["mIoU"] > best["mIoU"]:
                best = dict(m, iter=it + 1)
                torch.save(model.state_dict(), os.path.join(run_dir, "best.pt"))
            final = dict(m, iter=it + 1)
        if (it + 1) % 50 == 0 or miou != "":
            logw.writerow([it + 1, round(running, 4), opt.param_groups[0]["lr"], miou, round(time.time() - t0)]); log.flush()
    log.close()

    # We report the FINAL checkpoint (picking the best one on the test set would be cheating);
    # the best is only kept for reference.
    result = {"run": args.run, "mode": args.mode, "subset": subset_name, "n_labelled": len(train_ds),
              "seed": args.seed, "iters": args.iters, "final": final, "best_on_test_for_reference": best,
              "minutes": round((time.time() - t0) / 60, 1)}
    with open(os.path.join(run_dir, "metrics.json"), "w") as fh:
        json.dump(result, fh, indent=1)
    torch.save(model.state_dict(), os.path.join(run_dir, "final.pt"))
    res_path = os.path.join(args.out, "results.csv")
    new = not os.path.exists(res_path)
    with open(res_path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["run", "mode", "subset", "n_labelled", "seed", "iters", "mIoU", "pixel_acc", "mean_acc", "minutes"])
        w.writerow([args.run, args.mode, subset_name, len(train_ds), args.seed, args.iters,
                    round(final["mIoU"], 2), round(final["pixel_acc"], 2), round(final["mean_acc"], 2), result["minutes"]])
    print(f"FINAL {args.run}: mIoU {final['mIoU']:.2f}  pixel acc {final['pixel_acc']:.2f}  mean acc {final['mean_acc']:.2f}")


if __name__ == "__main__":
    main()
