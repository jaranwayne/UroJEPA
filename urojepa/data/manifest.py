# Copyright (c) 2026 the UroJEPA authors.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""CSV manifests (pandas only, no torch).

Pre-training manifest (unlabelled):    patient_id, video_path[, num_frames, view]
Fine-tuning / evaluation CSV (labelled): patient_id, video_path, label[, num_frames, view, center]

`video_path` may be absolute or relative to `root`. `num_frames` (slices per video) is
optional but avoids decoding every video to find its length; add it with
`scripts/build_manifest.py`.
"""

import os

import numpy as np
import pandas as pd


def read_manifest(path, root=None, require_label=False):
    df = pd.read_csv(path)
    required = ["patient_id", "video_path"] + (["label"] if require_label else [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing column(s) {missing}; found {list(df.columns)}")
    df = df.copy()
    df["patient_id"] = df["patient_id"].astype(str)
    if root:
        df["video_path"] = [p if os.path.isabs(p) else os.path.join(root, p) for p in df["video_path"]]
    if require_label:
        df["label"] = df["label"].astype(int)
    if "num_frames" in df.columns:
        df["num_frames"] = df["num_frames"].astype(int)
    return df.reset_index(drop=True)


def subsample_patients(df, fraction, seed):
    """Keep a random `fraction` of patients, stratified by outcome (label-efficiency
    experiments). All videos of a kept patient are kept. At least one patient per class."""
    if fraction is None or fraction >= 1.0:
        return df
    patient_label = df.groupby("patient_id")["label"].max()
    rng = np.random.default_rng(seed)
    keep = []
    for _, ids in patient_label.groupby(patient_label):
        ids = np.array(sorted(ids.index))
        n = max(1, int(round(fraction * len(ids))))
        keep.extend(rng.choice(ids, size=n, replace=False).tolist())
    return df[df["patient_id"].isin(set(keep))].reset_index(drop=True)
