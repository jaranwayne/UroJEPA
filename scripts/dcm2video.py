# Copyright (c) 2026 the UroJEPA authors.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Convert CT DICOM series to MP4 videos (one frame per slice).

One series (a folder of .dcm slices) -> one video:

    python scripts/dcm2video.py --dicom-dir /dicom/P0001/axial --out data/videos/P0001/axial.mp4

Every series under a root folder -> videos with the same relative layout
(/dicom/P0001/axial/*.dcm -> data/videos/P0001/axial.mp4):

    python scripts/dcm2video.py --dicom-root /dicom --out-root data/videos --workers 8

Conversion parameters (window, rescaling, frame size, fps) come from
configs/dcm2video.yaml and can be set per scanner; see that file.
"""

import argparse
import copy
import csv
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np
import yaml

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "dcm2video.yaml")
DEFAULT_PARAMS = {"window_center": 50, "window_width": 180, "rescale": False, "size": 512, "fps": 10}


# ---------------------------------------------------------------------------- parameters
def load_config(path=None):
    """Return {'default': {...}, 'devices': {...}}; missing keys fall back to DEFAULT_PARAMS."""
    cfg = {}
    if path is not None:
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
    default = dict(DEFAULT_PARAMS, **(cfg.get("default") or {}))
    return {"default": default, "devices": cfg.get("devices") or {}}


def select_params(ds, config, device=None):
    """Parameters for the series whose first slice is `ds` -> (profile name, params)."""
    devices = config["devices"]
    if device is not None:
        if device not in devices:
            raise KeyError(f"Device profile {device!r} not in config (available: {sorted(devices)})")
        name = device
    else:
        name = "default"
        for key, prof in devices.items():
            rules = (prof or {}).get("match") or {}
            if rules and all(str(v).lower() in str(getattr(ds, tag, "")).lower() for tag, v in rules.items()):
                name = key
                break
    params = copy.deepcopy(config["default"])
    if name != "default":
        params.update({k: v for k, v in (devices[name] or {}).items() if k != "match"})
    return name, params


# ---------------------------------------------------------------------------- conversion
def windowing(img, wc, ww):
    lo, hi = wc - ww / 2, wc + ww / 2
    img = np.clip(img, lo, hi)
    return ((img - lo) / (hi - lo) * 255).astype(np.uint8)


def series_to_frames(datasets, params):
    """Sorted DICOM slices -> uint8 grey frames [T, size, size]."""
    frames = []
    for ds in datasets:
        img = ds.pixel_array.astype(np.int32)  # windowing below runs in float64, as in the original notebook
        if params["rescale"]:
            img = img * float(getattr(ds, "RescaleSlope", 1)) + float(getattr(ds, "RescaleIntercept", 0))
        img = windowing(img, float(params["window_center"]), float(params["window_width"]))
        size = int(params["size"])
        frames.append(cv2.resize(img, (size, size)))
    return np.stack(frames)


def write_video(frames, path, fps):
    """Write grey frames [T, H, W] as an MP4 (mp4v) with 3 identical channels."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    T, H, W = frames.shape
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (W, H))
    if not writer.isOpened():
        raise IOError(f"Cannot open video writer for {path}")
    for f in frames:
        writer.write(np.repeat(f[:, :, None], 3, axis=2))
    writer.release()


def sort_slices(datasets):
    """Order slices by InstanceNumber (files without it keep their file-name order, at the end)."""
    def key(item):
        i, ds = item
        n = getattr(ds, "InstanceNumber", None)
        return (0, int(n), i) if n not in (None, "") else (1, 0, i)

    return [ds for _, ds in sorted(enumerate(datasets), key=key)]


def read_series(dicom_dir, ext=".dcm"):
    """Read all slices of one series; raises if the folder holds several series."""
    import pydicom

    files = sorted(
        os.path.join(dicom_dir, f)
        for f in os.listdir(dicom_dir)
        if os.path.isfile(os.path.join(dicom_dir, f)) and f.lower().endswith(ext.lower())
    )
    datasets = []
    for f in files:
        try:
            ds = pydicom.dcmread(f, force=True)
        except Exception:
            continue
        if "PixelData" in ds:
            datasets.append(ds)
    if not datasets:
        raise ValueError(f"No DICOM images in {dicom_dir}")
    series = {str(getattr(ds, "SeriesInstanceUID", "")) for ds in datasets}
    if len(series) > 1:
        raise ValueError(f"{dicom_dir} contains {len(series)} series; put each series in its own folder")
    return sort_slices(datasets)


def convert_series(dicom_dir, out_path, config, device=None, ext=".dcm"):
    """Convert one series; returns a summary dict."""
    datasets = read_series(dicom_dir, ext)
    profile, params = select_params(datasets[0], config, device)
    intercept = float(getattr(datasets[0], "RescaleIntercept", 0) or 0)
    if intercept != 0 and not params["rescale"]:
        print(
            f"[warn] {dicom_dir}: RescaleIntercept={intercept:g} but rescale=false; the window is applied "
            "to stored values. Add a device profile with rescale: true if these are not HU.",
            file=sys.stderr,
        )
    frames = series_to_frames(datasets, params)
    write_video(frames, out_path, params["fps"])
    return {"dicom_dir": dicom_dir, "video_path": out_path, "num_frames": len(frames), "device": profile}


def find_series_dirs(root, ext=".dcm"):
    out = []
    for dirpath, _, files in os.walk(root):
        if any(f.lower().endswith(ext.lower()) for f in files):
            out.append(dirpath)
    return sorted(out)


def _convert_job(job):
    dicom_dir, out_path, config, device, ext, overwrite = job
    if os.path.exists(out_path) and not overwrite:
        return {"dicom_dir": dicom_dir, "video_path": out_path, "num_frames": "", "device": "", "status": "exists"}
    try:
        res = convert_series(dicom_dir, out_path, config, device, ext)
        res["status"] = "ok"
    except Exception as e:
        res = {"dicom_dir": dicom_dir, "video_path": out_path, "num_frames": "", "device": "", "status": f"error: {e}"}
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--dicom-dir", help="folder with the slices of one series")
    src.add_argument("--dicom-root", help="root folder; every sub-folder with DICOM files is one series")
    p.add_argument("--out", help="output .mp4 (with --dicom-dir)")
    p.add_argument("--out-root", help="output folder (with --dicom-root)")
    p.add_argument("--config", default=DEFAULT_CONFIG, help="YAML with default and per-device parameters")
    p.add_argument("--device", default=None, help="force a device profile from the config")
    p.add_argument("--ext", default=".dcm", help="slice file extension ('' = all files)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--overwrite", action="store_true", help="re-create videos that already exist")
    args = p.parse_args()

    config = load_config(args.config if os.path.exists(args.config) else None)

    if args.dicom_dir:
        if not args.out:
            p.error("--out is required with --dicom-dir")
        res = convert_series(args.dicom_dir, args.out, config, args.device, args.ext)
        print(f"{res['video_path']}: {res['num_frames']} frames (device profile: {res['device']})")
        return

    if not args.out_root:
        p.error("--out-root is required with --dicom-root")
    series_dirs = find_series_dirs(args.dicom_root, args.ext)
    if not series_dirs:
        p.error(f"No DICOM series found under {args.dicom_root}")
    jobs = []
    for d in series_dirs:
        rel = os.path.relpath(d, args.dicom_root)
        name = os.path.basename(os.path.abspath(args.dicom_root)) if rel == "." else rel
        jobs.append((d, os.path.join(args.out_root, name + ".mp4"), config, args.device, args.ext, args.overwrite))

    with Pool(max(1, args.workers)) as pool:
        results = pool.map(_convert_job, jobs, chunksize=1)

    os.makedirs(args.out_root, exist_ok=True)
    log_path = os.path.join(args.out_root, "conversion_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dicom_dir", "video_path", "num_frames", "device", "status"])
        writer.writeheader()
        writer.writerows(results)
    n_ok = sum(r["status"] == "ok" for r in results)
    n_err = sum(r["status"].startswith("error") for r in results)
    print(f"{n_ok} converted, {len(results) - n_ok - n_err} already present, {n_err} failed; log: {log_path}")


if __name__ == "__main__":
    main()
