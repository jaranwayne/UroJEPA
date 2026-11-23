# Copyright (c) 2026 the UroJEPA authors.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Build a CSV manifest of CT videos with the number of slices of each video.

From a directory laid out as <video-dir>/<patient_id>/<series>.mp4:

    python scripts/build_manifest.py --video-dir /data/ct_videos --output data/pretrain_manifest.csv \
        --exclude-patients data/all_outcome_and_reader_patients.csv --min-frames 16

From an existing CSV with patient_id,video_path[,label,...] (adds/refreshes num_frames,
keeps all other columns; use this for the fine-tuning label files too):

    python scripts/build_manifest.py --input data/stone_sepsis_labels.csv --output data/stone_sepsis_labels.csv

Prints the corpus size and the dual-scale clip inventory (Supplementary Fig. 4).
"""

import argparse
import os
import sys
from multiprocessing import Pool

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from urojepa.data.clips import clip_inventory  # noqa: E402
from urojepa.data.video_io import count_frames  # noqa: E402

VIDEO_EXT = (".mp4", ".avi", ".mkv", ".mov")


def _count(path):
    try:
        return count_frames(path)
    except Exception as e:  # unreadable file
        print(f"[warn] cannot read {path}: {e}", file=sys.stderr)
        return 0


def scan_directory(video_dir):
    rows = []
    for dirpath, _, files in os.walk(video_dir):
        for fn in sorted(files):
            if fn.lower().endswith(VIDEO_EXT):
                path = os.path.join(dirpath, fn)
                rel = os.path.relpath(path, video_dir)
                parts = rel.split(os.sep)
                patient = parts[0] if len(parts) > 1 else os.path.splitext(fn)[0]
                low = fn.lower()
                view = "axial" if "axial" in low else ("coronal" if "coronal" in low else "")
                rows.append({"patient_id": patient, "video_path": path, "view": view})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video-dir", help="directory with <patient_id>/<series>.mp4")
    src.add_argument("--input", help="existing CSV with patient_id,video_path")
    p.add_argument("--output", required=True)
    p.add_argument("--root", default=None, help="prefix for relative video_path values in --input")
    p.add_argument("--exclude-patients", default=None, help="CSV with a patient_id column to exclude")
    p.add_argument("--min-frames", type=int, default=0, help="drop videos with fewer slices (16 for pre-training)")
    p.add_argument("--num-frames", type=int, default=16, help="frames per clip, for the inventory")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    if args.video_dir:
        df = scan_directory(args.video_dir)
    else:
        df = pd.read_csv(args.input)
        assert {"patient_id", "video_path"} <= set(df.columns), "need patient_id and video_path columns"
    df["patient_id"] = df["patient_id"].astype(str)
    paths = [
        vp if (args.root is None or os.path.isabs(vp)) else os.path.join(args.root, vp) for vp in df["video_path"]
    ]

    if args.exclude_patients:
        excl = set(pd.read_csv(args.exclude_patients)["patient_id"].astype(str))
        keep = ~df["patient_id"].isin(excl)
        print(f"Excluding {int((~keep).sum())} videos of {len(excl)} listed patients")
        df, paths = df[keep].reset_index(drop=True), [pth for pth, k in zip(paths, keep) if k]

    with Pool(args.workers) as pool:
        df["num_frames"] = pool.map(_count, paths, chunksize=8)

    n_bad = int((df["num_frames"] < max(1, args.min_frames)).sum())
    if n_bad:
        print(f"Dropping {n_bad} unreadable videos or videos with < {args.min_frames} slices")
        df = df[df["num_frames"] >= max(1, args.min_frames)].reset_index(drop=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df.to_csv(args.output, index=False)

    inv = clip_inventory(df["num_frames"].values, args.num_frames, (1, 2))
    print(
        f"Wrote {args.output}: {df.patient_id.nunique()} patients, {len(df)} videos, "
        f"{int(df.num_frames.sum())} frames"
    )
    print(f"Candidate {args.num_frames}-frame clips: local (d=1) {inv[1]}, broader (d=2) {inv[2]}, total {sum(inv.values())}")


if __name__ == "__main__":
    main()
