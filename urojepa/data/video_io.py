# Copyright (c) 2026 the UroJEPA authors.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Reading CT videos (one frame per slice). Uses decord when installed (as V-JEPA 2)
and falls back to OpenCV. Set UROJEPA_VIDEO_BACKEND=cv2|decord to force a backend."""

import os

import numpy as np

_BACKEND = os.environ.get("UROJEPA_VIDEO_BACKEND", "").lower()
try:
    if _BACKEND in ("", "decord"):
        from decord import VideoReader, cpu  # noqa: F401

        _BACKEND = "decord"
except ImportError:
    _BACKEND = "cv2"
if _BACKEND not in ("decord", "cv2"):
    raise ValueError(f"Unknown UROJEPA_VIDEO_BACKEND {_BACKEND!r}")


def get_backend():
    return _BACKEND


def load_video(path, indices=None):
    """Return (frames, length): frames is a uint8 RGB array [len(indices), H, W, 3]
    (all frames if indices is None) and length is the number of frames in the video."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    if _BACKEND == "decord":
        from decord import VideoReader, cpu

        vr = VideoReader(path, num_threads=1, ctx=cpu(0))
        length = len(vr)
        if indices is None:
            indices = np.arange(length)
        indices = np.asarray(indices, dtype=np.int64)
        if len(indices) and indices.max() >= length:
            raise IndexError(f"{path}: requested frame {indices.max()} but video has {length} frames")
        return vr.get_batch(list(indices)).asnumpy(), length

    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {path}")
    try:
        stop = None if indices is None else int(np.max(indices)) + 1
        frames = []
        while stop is None or len(frames) < stop:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    if len(frames) == 0:
        raise IOError(f"No frames decoded from {path}")
    if indices is None:
        out = np.stack(frames)
        length = len(frames)
    else:
        indices = np.asarray(indices, dtype=np.int64)
        if indices.max() >= len(frames):
            raise IndexError(f"{path}: requested frame {indices.max()} but only {len(frames)} decoded")
        out = np.stack([frames[i] for i in indices])
        length = None  # unknown without decoding the full video
    return out[..., ::-1].copy(), length  # BGR -> RGB


def count_frames(path):
    """Exact number of decodable frames (slices)."""
    if _BACKEND == "decord":
        from decord import VideoReader, cpu

        return len(VideoReader(path, num_threads=1, ctx=cpu(0)))
    import cv2

    cap = cv2.VideoCapture(path)
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return n
