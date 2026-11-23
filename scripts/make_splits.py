# Copyright (c) 2026 the UroJEPA authors.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Outcome-stratified train/validation/test split (7:1:2) of a labelled task cohort.

    python scripts/make_splits.py --labels data/stone_sepsis_labels.csv --out data/splits/stone_sepsis

As in the paper, the default split is made at the VIDEO level, so the axial and
coronal videos of one patient may fall into different sets. Use --group-by-patient
to keep all videos of a patient in the same set (no patient overlap).
The split is fixed by --seed and shared by all models and fine-tuning seeds.
"""

import argparse
import os

import numpy as np
import pandas as pd


def stratified_split(df, ratios, seed, group_by_patient=False):
    rng = np.random.default_rng(seed)
    if group_by_patient:
        units = df.groupby("patient_id")["label"].max().reset_index()
        key = "patient_id"
    else:
        units = df.reset_index().rename(columns={"index": "_row"})[["_row", "label"]]
        key = "_row"

    assign = {}
    for _, g in units.groupby("label"):
        ids = g[key].values.copy()
        rng.shuffle(ids)
        n = len(ids)
        n_test = int(round(ratios[2] * n))
        n_val = int(round(ratios[1] * n))
        for i, u in enumerate(ids):
            assign[u] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")

    split = df["patient_id"].map(assign) if group_by_patient else pd.Series(range(len(df))).map(assign).values
    return {s: df[np.asarray(split) == s].reset_index(drop=True) for s in ("train", "val", "test")}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--labels", required=True, help="CSV with patient_id,video_path,label[,num_frames,view]")
    p.add_argument("--out", required=True, help="output directory for train/val/test.csv")
    p.add_argument("--ratios", type=float, nargs=3, default=[0.7, 0.1, 0.2])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--group-by-patient", action="store_true")
    args = p.parse_args()

    df = pd.read_csv(args.labels)
    df["patient_id"] = df["patient_id"].astype(str)
    df["label"] = df["label"].astype(int)
    assert abs(sum(args.ratios) - 1.0) < 1e-6, "ratios must sum to 1"

    splits = stratified_split(df, args.ratios, args.seed, args.group_by_patient)
    os.makedirs(args.out, exist_ok=True)
    for name, d in splits.items():
        d.to_csv(os.path.join(args.out, f"{name}.csv"), index=False)
        print(
            f"{name:5s}: {len(d):5d} videos, {d.patient_id.nunique():5d} patients, "
            f"{int(d.label.sum()):4d} positive videos ({d.label.mean():.1%})"
        )
    shared = set(splits["train"].patient_id) & set(splits["test"].patient_id)
    print(f"Patients with videos in both train and test: {len(shared)}")


if __name__ == "__main__":
    main()
