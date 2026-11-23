# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Adapted from V-JEPA 2 (src/masks/multiseq_multiblock3d.py).
# Changes for UroJEPA: the dataset yields plain clip tensors [C, T, H, W]; unused
# masking variants (inverse block, full complement) were removed.

import math
from multiprocessing import Value

import torch


class MaskCollator(object):
    """Collates clips into a batch and samples spatio-temporal multi-block masks.

    Returns a list with one entry per clip length present in the batch:
        (clips [B, C, T, H, W], masks_enc [n_gen][B, N_ctx], masks_pred [n_gen][B, N_pred])
    """

    def __init__(self, cfgs_mask, dataset_fpcs, crop_size=224, patch_size=16, tubelet_size=2):
        self.mask_generators = dict()
        for fpc in dataset_fpcs:
            self.mask_generators[fpc] = [
                _MaskGenerator(
                    crop_size=crop_size,
                    num_frames=fpc,
                    spatial_patch_size=patch_size,
                    temporal_patch_size=tubelet_size,
                    spatial_pred_mask_scale=m.get("spatial_scale"),
                    temporal_pred_mask_scale=m.get("temporal_scale"),
                    aspect_ratio=m.get("aspect_ratio"),
                    npred=m.get("num_blocks"),
                    max_context_frames_ratio=m.get("max_temporal_keep", 1.0),
                    max_keep=m.get("max_keep", None),
                )
                for m in cfgs_mask
            ]

    def step(self):
        for fpc in self.mask_generators:
            for mask_generator in self.mask_generators[fpc]:
                mask_generator.step()

    def __call__(self, batch):
        filtered = {fpc: [] for fpc in self.mask_generators}
        for clip in batch:
            fpc = clip.shape[1]
            if fpc not in filtered:
                raise ValueError(f"Clip with {fpc} frames but no mask generator for it")
            filtered[fpc].append(clip)

        collations = []
        for fpc, clips in filtered.items():
            if len(clips) == 0:
                continue
            clips = torch.stack(clips, dim=0)
            masks_enc, masks_pred = [], []
            for mask_generator in self.mask_generators[fpc]:
                m_enc, m_pred = mask_generator(len(clips))
                masks_enc.append(m_enc)
                masks_pred.append(m_pred)
            collations.append((clips, masks_enc, masks_pred))
        return collations


class _MaskGenerator(object):

    def __init__(
        self,
        crop_size=224,
        num_frames=16,
        spatial_patch_size=16,
        temporal_patch_size=2,
        spatial_pred_mask_scale=(0.2, 0.8),
        temporal_pred_mask_scale=(1.0, 1.0),
        aspect_ratio=(0.3, 3.0),
        npred=1,
        max_context_frames_ratio=1.0,
        max_keep=None,
    ):
        if not isinstance(crop_size, (tuple, list)):
            crop_size = (crop_size,) * 2
        if not isinstance(spatial_patch_size, (tuple, list)):
            spatial_patch_size = (spatial_patch_size,) * 2
        self.height, self.width = [crop_size[i] // spatial_patch_size[i] for i in (0, 1)]
        self.duration = num_frames // temporal_patch_size

        self.aspect_ratio = aspect_ratio
        self.spatial_pred_mask_scale = spatial_pred_mask_scale
        self.temporal_pred_mask_scale = temporal_pred_mask_scale
        self.npred = npred
        self.max_context_duration = max(1, int(self.duration * max_context_frames_ratio))
        self.max_keep = max_keep
        self._itr_counter = Value("i", -1)  # shared across dataloader workers

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_block_size(self, generator, temporal_scale, spatial_scale, aspect_ratio_scale):
        _rand = torch.rand(1, generator=generator).item()
        min_t, max_t = temporal_scale
        t = max(1, int(self.duration * (min_t + _rand * (max_t - min_t))))

        _rand = torch.rand(1, generator=generator).item()
        min_s, max_s = spatial_scale
        spatial_num_keep = int(self.height * self.width * (min_s + _rand * (max_s - min_s)))

        _rand = torch.rand(1, generator=generator).item()
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)

        h = min(int(round(math.sqrt(spatial_num_keep * aspect_ratio))), self.height)
        w = min(int(round(math.sqrt(spatial_num_keep / aspect_ratio))), self.width)
        return (t, h, w)

    def _sample_block_mask(self, b_size):
        t, h, w = b_size
        top = torch.randint(0, self.height - h + 1, (1,))
        left = torch.randint(0, self.width - w + 1, (1,))
        start = torch.randint(0, self.duration - t + 1, (1,))

        mask = torch.ones((self.duration, self.height, self.width), dtype=torch.int32)
        mask[start : start + t, top : top + h, left : left + w] = 0
        if self.max_context_duration < self.duration:
            mask[self.max_context_duration :, :, :] = 0
        return mask

    def __call__(self, batch_size):
        """1) sample the block size (shared across the batch, seeded by the step counter);
        2) sample block locations per sample; 3) return context (enc) and target (pred) indices."""
        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        p_size = self._sample_block_size(
            generator=g,
            temporal_scale=self.temporal_pred_mask_scale,
            spatial_scale=self.spatial_pred_mask_scale,
            aspect_ratio_scale=self.aspect_ratio,
        )

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_enc = min_keep_pred = self.duration * self.height * self.width
        for _ in range(batch_size):
            empty_context = True
            while empty_context:
                mask_e = torch.ones((self.duration, self.height, self.width), dtype=torch.int32)
                for _ in range(self.npred):
                    mask_e *= self._sample_block_mask(p_size)
                mask_e = mask_e.flatten()
                mask_p = torch.argwhere(mask_e == 0).squeeze(-1)
                mask_e = torch.nonzero(mask_e).squeeze(-1)
                empty_context = len(mask_e) == 0
                if not empty_context:
                    min_keep_pred = min(min_keep_pred, len(mask_p))
                    min_keep_enc = min(min_keep_enc, len(mask_e))
                    collated_masks_pred.append(mask_p)
                    collated_masks_enc.append(mask_e)

        if self.max_keep is not None:
            min_keep_enc = min(min_keep_enc, self.max_keep)

        collated_masks_enc = torch.stack([cm[:min_keep_enc] for cm in collated_masks_enc])
        collated_masks_pred = torch.stack([cm[:min_keep_pred] for cm in collated_masks_pred])
        return collated_masks_enc, collated_masks_pred
