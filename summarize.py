"""Print a mean ± std table from runs/results.csv, grouped by (mode, labelled fraction, method).

  python summarize.py            # reads runs/results.csv
  python summarize.py --csv path/to/results.csv --md table.md
"""
import argparse
import csv
import re
from collections import defaultdict

import numpy as np


def group_key(row):
    s = row["subset"]
    m = re.match(r"(?P<method>.+)_frac(?P<frac>[\d.]+)_seed\d+", s)
    if m:
        return row["mode"], f"{float(m['frac']) * 100:.0f}%", m["method"]
    return row["mode"], "100%", "all"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="runs/results.csv")
    ap.add_argument("--md", help="also write the table to this markdown file")
    args = ap.parse_args()
    groups = defaultdict(list)
    with open(args.csv) as fh:
        for row in csv.DictReader(fh):
            groups[group_key(row)].append(row)
    lines = ["| input | labels | selection | runs (seeds) | mIoU (mean ± std) | pixel acc | n images |",
             "|---|---|---|---|---|---|---|"]
    for key in sorted(groups, key=lambda k: (k[0], float(k[1][:-1]), k[2])):
        rows = groups[key]
        miou = np.array([float(r["mIoU"]) for r in rows])
        pa = np.array([float(r["pixel_acc"]) for r in rows])
        seeds = ",".join(sorted({r["seed"] for r in rows}))
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {len(rows)} ({seeds}) | "
                     f"{miou.mean():.2f} ± {miou.std(ddof=1) if len(rows) > 1 else 0:.2f} | "
                     f"{pa.mean():.2f} | {rows[0]['n_labelled']} |")
    table = "\n".join(lines)
    print(table)
    if args.md:
        with open(args.md, "w") as fh:
            fh.write(table + "\n")


if __name__ == "__main__":
    main()
