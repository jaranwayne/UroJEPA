# Copyright (c) 2026 the UroJEPA authors.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from urojepa.data.clips import eval_clip_indices, sample_train_clip
from urojepa.data.video_io import load_video
from urojepa.utils.logging import get_logger

logger = get_logger(__name__)


class CTClassificationDataset(Dataset):
    """Labelled CT videos for outcome prediction.

    training=True : one random 16-slice clip per video (scale and start drawn as in
                    pre-training); returns (clip [C, T, H, W], label, index)
    training=False: `eval_num_clips` evenly spaced clips; returns
                    (clips [K, C, T, H, W], label, index); logits are averaged over K.
    """

    def __init__(
        self,
        df,
        training,
        transform,
        num_frames=16,
        frame_steps=(1, 2),
        eval_num_clips=9,
        eval_frame_step=1,
    ):
        self.df = df.reset_index(drop=True)
        self.paths = self.df["video_path"].tolist()
        self.labels = self.df["label"].astype(int).tolist()
        self.lengths = self.df["num_frames"].tolist() if "num_frames" in self.df.columns else None
        self.training = training
        self.transform = transform
        self.num_frames = num_frames
        self.frame_steps = tuple(frame_steps)
        self.eval_num_clips = eval_num_clips
        self.eval_frame_step = eval_frame_step

    def __len__(self):
        return len(self.paths)

    def _clip_indices(self, length):
        if self.training:
            rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,)).item()))
            return [sample_train_clip(length, rng, self.num_frames, self.frame_steps)]
        return eval_clip_indices(length, self.eval_num_clips, self.num_frames, self.eval_frame_step)

    def __getitem__(self, index):
        path = self.paths[index]
        if self.lengths is not None:
            clips_idx = self._clip_indices(int(self.lengths[index]))
            needed = np.unique(np.concatenate(clips_idx))
            frames, _ = load_video(path, needed)
            lookup = {int(f): i for i, f in enumerate(needed)}
            clips = [frames[[lookup[int(f)] for f in ci]] for ci in clips_idx]
        else:
            frames, length = load_video(path, None)
            clips = [frames[ci] for ci in self._clip_indices(length)]

        clips = [self.transform(c) for c in clips]
        label = self.labels[index]
        if self.training:
            return clips[0], label, index
        return torch.stack(clips, dim=0), label, index


def make_finetune_loader(dataset, batch_size, training, rank=0, world_size=1, seed=0, num_workers=8, pin_mem=True):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=training, seed=seed)
    loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        drop_last=training and len(dataset) >= batch_size * world_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=False,
    )
    return loader, sampler
