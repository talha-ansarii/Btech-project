"""Run the whole baseline plan one run after another, skipping runs that already finished.

Safe to re-run after Colab/Kaggle disconnects: finished runs (with metrics.json) are skipped,
an interrupted run starts again from scratch.

  python run_baseline.py --data data/nyuv2 --out /content/drive/MyDrive/rgbx/runs            # step A: RGB-D
  python run_baseline.py --data data/nyuv2 --out ... --modes rgb                             # step B: RGB-only
  python run_baseline.py --data data/nyuv2 --out ... --only full                             # just the full-data run
  python run_baseline.py ... --dry-run                                                       # print the commands only
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def plan(modes, seeds, full_iters, low_iters):
    runs = []
    for mode in modes:
        runs.append((f"full_{mode}_s0", mode, None, 0, full_iters))
        for frac in ("0.05", "0.10"):
            for s in seeds:
                pct = int(round(float(frac) * 100))
                runs.append((f"rand{pct}_{mode}_s{s}", mode,
                             os.path.join(HERE, "splits", f"random_frac{frac}_seed{s}.json"), s, low_iters))
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--modes", nargs="+", default=["rgbd"], choices=["rgbd", "rgb"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--full-iters", type=int, default=10000)
    ap.add_argument("--low-iters", type=int, default=4000)
    ap.add_argument("--only", choices=["full", "low"], help="run only the full-data or only the low-label runs")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("extra", nargs=argparse.REMAINDER, help="anything after -- is passed to train.py")
    args = ap.parse_args()
    extra = args.extra[1:] if args.extra[:1] == ["--"] else args.extra

    for name, mode, subset, seed, iters in plan(args.modes, args.seeds, args.full_iters, args.low_iters):
        if args.only == "full" and subset or args.only == "low" and not subset:
            continue
        if os.path.exists(os.path.join(args.out, name, "metrics.json")):
            print(f"skip {name} (already finished)")
            continue
        cmd = [sys.executable, os.path.join(HERE, "train.py"), "--data", args.data, "--out", args.out,
               "--run", name, "--mode", mode, "--seed", str(seed), "--iters", str(iters)]
        if subset:
            cmd += ["--subset", subset]
        cmd += extra
        print(" ".join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True)
    if not args.dry_run:
        subprocess.run([sys.executable, os.path.join(HERE, "summarize.py"), "--csv", os.path.join(args.out, "results.csv"),
                        "--md", os.path.join(args.out, "summary.md")])


if __name__ == "__main__":
    main()
