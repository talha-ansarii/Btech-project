"""Train one run per (selection method, budget, seed), skipping runs that already finished.

Same settings as the random baseline (4000 steps, batch 8, ...), and results go to the SAME
results.csv, so summarize.py puts every method in one table next to random.

  python run_selection.py --data data/nyuv2 --out $RUNS --splits-dir $SPLITS \
         --jobs cdiv:0.05 kmeans:0.05 r2-entropy:0.10 r2-coreset:0.10 r2-cdiv:0.10 r2-mcomp:0.10
  python run_selection.py ... --dry-run        # print the commands only
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--splits-dir", required=True)
    ap.add_argument("--jobs", nargs="+", required=True, help="method:fraction, e.g. cdiv:0.05")
    ap.add_argument("--modes", nargs="+", default=["rgbd"], choices=["rgbd", "rgb"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("extra", nargs=argparse.REMAINDER, help="anything after -- is passed to train.py")
    args = ap.parse_args()
    extra = args.extra[1:] if args.extra[:1] == ["--"] else args.extra

    for job in args.jobs:
        method, frac = job.split(":")
        frac = float(frac)
        for mode in args.modes:
            for s in args.seeds:
                name = f"{method}{int(round(frac * 100))}_{mode}_s{s}"
                split = os.path.join(args.splits_dir, f"{method}_frac{frac:.2f}_seed{s}.json")
                if os.path.exists(os.path.join(args.out, name, "metrics.json")):
                    print(f"skip {name} (already finished)")
                    continue
                if not os.path.exists(split):
                    print(f"MISSING {split}: run select_subsets.py first; skipping {name}")
                    continue
                cmd = [sys.executable, os.path.join(HERE, "train.py"), "--data", args.data, "--out", args.out,
                       "--run", name, "--mode", mode, "--seed", str(s), "--iters", str(args.iters),
                       "--subset", split] + extra
                print(" ".join(cmd), flush=True)
                if not args.dry_run:
                    subprocess.run(cmd, check=True)
    if not args.dry_run:
        subprocess.run([sys.executable, os.path.join(HERE, "summarize.py"), "--csv", os.path.join(args.out, "results.csv"),
                        "--md", os.path.join(args.out, "summary.md")])


if __name__ == "__main__":
    main()
