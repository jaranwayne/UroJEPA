# Copyright (c) 2026 the UroJEPA authors.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Collect metrics.json files written by finetune.py into one table
(mean ± SD over seeds per task, model, label fraction and cohort).

    python scripts/summarize.py --root outputs/finetune --out outputs/summary.csv
Expected layout: <root>/<task>/<model>/[frac<f>/]seed<k>/metrics.json
"""

import argparse
import glob
import json
import os

import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--root", default="outputs/finetune")
    p.add_argument("--out", default=None, help="optional CSV with one row per run")
    p.add_argument("--metric", default="auroc")
    args = p.parse_args()

    rows = []
    for path in glob.glob(os.path.join(args.root, "**", "metrics.json"), recursive=True):
        parts = os.path.relpath(path, args.root).split(os.sep)[:-1]
        seed_dir = parts[-1]
        frac = 1.0
        if len(parts) >= 2 and parts[-2].startswith("frac"):
            frac = float(parts[-2][4:])
            parts = parts[:-1]
        task, model = (parts[0], "/".join(parts[1:-1])) if len(parts) > 2 else (parts[0], "")
        with open(path) as f:
            m = json.load(f)
        for cohort, res in m.items():
            if isinstance(res, dict) and args.metric in res:
                rows.append(
                    {
                        "task": task,
                        "model": model,
                        "train_fraction": frac,
                        "seed": seed_dir,
                        "cohort": cohort,
                        args.metric: res[args.metric],
                    }
                )
    if not rows:
        print(f"No metrics.json found under {args.root}")
        return
    df = pd.DataFrame(rows)
    if args.out:
        df.to_csv(args.out, index=False)
    summary = (
        df.groupby(["task", "model", "train_fraction", "cohort"])[args.metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
