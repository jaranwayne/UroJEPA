# Copyright (c) 2026 the UroJEPA authors.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from urojepa.data.clips import clip_indices, clip_inventory, draw_hierarchical_clips, num_clips
from urojepa.data.manifest import read_manifest
from urojepa.data.video_io import count_frames, load_video
from urojepa.utils.logging import get_logger

logger = get_logger(__name__)


class CTPretrainDataset(Dataset):
    """Unlabelled CT videos. Indexed by (video_idx, frame_step, start) tuples produced by
    `HierarchicalClipSampler`; returns one transformed clip [C, T, H, W]."""

    def __init__(self, manifest, root=None, num_frames=16, frame_steps=(1, 2), transform=None, max_retries=10):
        df = read_manifest(manifest, root=root) if isinstance(manifest, str) else manifest.reset_index(drop=True)
        if "num_frames" not in df.columns:
            logger.info("Manifest has no `num_frames` column; counting frames (run scripts/build_manifest.py to cache)")
            df["num_frames"] = [count_frames(p) for p in df["video_path"]]

        min_len = num_frames  # at least one local (d = min step) clip; no padding by repetition
        n_before = len(df)
        df = df[num_clips(df["num_frames"].values, num_frames, min(frame_steps)) > 0].reset_index(drop=True)
        if len(df) < n_before:
            logger.info(f"Dropped {n_before - len(df)} videos with fewer than {min_len} slices")

        self.df = df
        self.paths = df["video_path"].tolist()
        self.lengths = df["num_frames"].values.astype(np.int64)
        self.num_frames = num_frames
        self.frame_steps = tuple(frame_steps)
        self.transform = transform
        self.max_retries = max_retries

        groups = df.groupby("patient_id", sort=True).indices
        self.patient_videos = [np.asarray(v, dtype=np.int64) for v in groups.values()]
        inv = clip_inventory(self.lengths, num_frames, self.frame_steps)
        self.num_candidate_clips = int(sum(inv.values()))
        logger.info(
            f"Pre-training corpus: {len(self.patient_videos)} patients, {len(df)} videos, "
            f"{int(self.lengths.sum())} frames, candidate clips per frame step {inv} "
            f"(total {self.num_candidate_clips})"
        )

    def __len__(self):
        return self.num_candidate_clips

    def _load(self, video_idx, frame_step, start):
        idx = clip_indices(start, self.num_frames, frame_step)
        frames, _ = load_video(self.paths[video_idx], idx)
        return self.transform(frames) if self.transform is not None else frames

    def __getitem__(self, key):
        video_idx, frame_step, start = key
        for attempt in range(self.max_retries):
            try:
                return self._load(video_idx, frame_step, start)
            except Exception as e:  # unreadable/corrupt video: draw another clip
                logger.warning(f"Failed to load {self.paths[video_idx]} (attempt {attempt}): {e}")
                rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,)).item()))
                v, d, s = draw_hierarchical_clips(
                    self.patient_videos, self.lengths, 1, rng, self.num_frames, self.frame_steps
                )
                video_idx, frame_step, start = int(v[0]), int(d[0]), int(s[0])
        raise RuntimeError(f"Could not load a clip after {self.max_retries} attempts")


class HierarchicalClipSampler(Sampler):
    """Distributed sampler implementing patient -> video -> scale -> clip sampling.

    All ranks draw the same global sequence (seeded by seed and epoch) and keep every
    world_size-th element, so ranks see disjoint draws.
    """

    def __init__(self, dataset, samples_per_epoch, rank=0, world_size=1, seed=0):
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0
        self.num_samples = int(samples_per_epoch) // world_size

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        rng = np.random.default_rng([self.seed, self.epoch])
        total = self.num_samples * self.world_size
        v, d, s = draw_hierarchical_clips(
            self.dataset.patient_videos,
            self.dataset.lengths,
            total,
            rng,
            self.dataset.num_frames,
            self.dataset.frame_steps,
        )
        for i in range(self.rank, total, self.world_size):
            yield (int(v[i]), int(d[i]), int(s[i]))


def make_pretrain_loader(
    dataset,
    batch_size,
    collator,
    samples_per_epoch=None,
    rank=0,
    world_size=1,
    seed=0,
    num_workers=8,
    pin_mem=True,
    persistent_workers=True,
):
    if samples_per_epoch is None:
        samples_per_epoch = len(dataset)  # one "epoch" = as many draws as candidate clips
    sampler = HierarchicalClipSampler(dataset, samples_per_epoch, rank=rank, world_size=world_size, seed=seed)
    loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        collate_fn=collator,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=(num_workers > 0) and persistent_workers,
    )
    return loader, sampler
