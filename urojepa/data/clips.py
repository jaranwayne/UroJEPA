# Copyright (c) 2026 the UroJEPA authors.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Dual-scale clip construction for CT videos (Supplementary methods).

A CT video of L slices (5-mm spacing) is cut into clips of T = 16 ordered slices
    I(s, d) = (s, s + d, ..., s + (T - 1) d),   0 <= s <= L - 1 - (T - 1) d,
with d = 1 (local scale: 5-mm spacing, 75-mm span) or d = 2 (broader scale: 10-mm
spacing, 150-mm span). Each video yields N(L, d) = max(0, L - (T - 1) d) clips per
scale; videos with L < 31 contribute local clips only, and videos are never padded
by repetition during pre-training.

During pre-training a clip is drawn hierarchically: patient (uniform) -> video of
that patient (uniform) -> scale (uniform if both exist) -> clip start (uniform).

This module is numpy-only so that it can be unit-tested without torch.
"""

import numpy as np


def num_clips(length, num_frames=16, frame_step=1):
    """N(L, d) = max(0, L - (T - 1) d)."""
    return np.maximum(0, np.asarray(length) - (num_frames - 1) * frame_step)


def clip_indices(start, num_frames=16, frame_step=1):
    """I(s, d) = (s, s + d, ..., s + (T - 1) d)."""
    return int(start) + int(frame_step) * np.arange(num_frames, dtype=np.int64)


def clip_inventory(lengths, num_frames=16, frame_steps=(1, 2)):
    """Number of candidate clips per scale over a set of videos (Supplementary Fig. 4b)."""
    lengths = np.asarray(lengths)
    return {int(d): int(num_clips(lengths, num_frames, d).sum()) for d in frame_steps}


def draw_hierarchical_clips(patient_videos, video_lengths, n, rng, num_frames=16, frame_steps=(1, 2)):
    """Draw `n` clips with patient -> video -> scale -> clip sampling.

    :param patient_videos: list (one entry per patient) of arrays of video indices; every
        listed video must have at least one valid clip at the smallest frame step.
    :param video_lengths: array with the number of slices of every video
    :param rng: numpy.random.Generator
    :returns: (video_idx, frame_step, start) integer arrays of length n
    """
    video_lengths = np.asarray(video_lengths)
    steps = np.asarray(frame_steps, dtype=np.int64)
    counts = np.array([len(v) for v in patient_videos], dtype=np.int64)
    if (counts == 0).any():
        raise ValueError("Every patient must have at least one usable video")
    offsets = np.concatenate([[0], np.cumsum(counts)[:-1]])
    flat = np.concatenate([np.asarray(v, dtype=np.int64) for v in patient_videos])

    patient = rng.integers(0, len(patient_videos), size=n)
    k = np.floor(rng.random(n) * counts[patient]).astype(np.int64)
    video = flat[offsets[patient] + k]

    nclips = num_clips(video_lengths[video][:, None], num_frames, steps[None, :])  # [n, S]
    valid = nclips > 0
    n_valid = valid.sum(axis=1)
    if (n_valid == 0).any():
        raise ValueError("A sampled video has no valid clip; filter the manifest first")
    choice = np.floor(rng.random(n) * n_valid).astype(np.int64)
    rank_among_valid = np.cumsum(valid, axis=1) - 1
    scale_idx = np.argmax(valid & (rank_among_valid == choice[:, None]), axis=1)

    step = steps[scale_idx]
    n_start = nclips[np.arange(n), scale_idx]
    start = np.floor(rng.random(n) * n_start).astype(np.int64)
    return video, step, start


def sample_train_clip(length, rng, num_frames=16, frame_steps=(1, 2)):
    """Random clip of one video for fine-tuning (same scale/start rule as pre-training).

    Videos shorter than `num_frames` are padded by repeating the last slice so that
    every labelled video can be used.
    """
    steps = [d for d in frame_steps if num_clips(length, num_frames, d) > 0]
    if not steps:
        return np.minimum(np.arange(num_frames, dtype=np.int64), max(0, length - 1))
    d = steps[int(rng.integers(0, len(steps)))]
    s = int(rng.integers(0, int(num_clips(length, num_frames, d))))
    return clip_indices(s, num_frames, d)


def eval_clip_indices(length, num_clips_per_video=9, num_frames=16, frame_step=1):
    """Deterministic test-time clips: `num_clips_per_video` starts evenly spaced over the
    video (logits are averaged over them). Falls back to d = 1 if the video is too short
    for `frame_step`, and pads by repeating the last slice if L < num_frames."""
    if num_clips(length, num_frames, frame_step) <= 0:
        frame_step = 1
    if length < num_frames:
        idx = np.minimum(np.arange(num_frames, dtype=np.int64), max(0, length - 1))
        return [idx.copy() for _ in range(num_clips_per_video)]
    max_start = length - 1 - (num_frames - 1) * frame_step
    starts = np.round(np.linspace(0, max_start, num_clips_per_video)).astype(np.int64)
    return [clip_indices(s, num_frames, frame_step) for s in starts]
