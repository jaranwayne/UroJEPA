# Copyright (c) 2026 the UroJEPA authors.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Clip-level transforms. Every random parameter is drawn once per clip, so the same
spatial and intensity augmentation is applied to all slices of a clip.

Random numbers come from torch, which DataLoader seeds independently per worker.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _uniform(lo, hi):
    return float(lo + (hi - lo) * torch.rand(1).item())


def _random_resized_crop_params(H, W, scale, ratio):
    area = H * W
    log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
    for _ in range(10):
        target_area = area * _uniform(*scale)
        aspect = math.exp(_uniform(*log_ratio))
        w = int(round(math.sqrt(target_area * aspect)))
        h = int(round(math.sqrt(target_area / aspect)))
        if 0 < w <= W and 0 < h <= H:
            i = int(torch.randint(0, H - h + 1, (1,)).item())
            j = int(torch.randint(0, W - w + 1, (1,)).item())
            return i, j, h, w
    # Fallback: central crop
    in_ratio = W / H
    if in_ratio < ratio[0]:
        w, h = W, int(round(W / ratio[0]))
    elif in_ratio > ratio[1]:
        h, w = H, int(round(H * ratio[1]))
    else:
        w, h = W, H
    return (H - h) // 2, (W - w) // 2, h, w


class VideoTransform:
    """uint8 frames [T, H, W, 3] -> normalised float clip [3, T, crop, crop].

    training=True : random resized crop, horizontal flip, brightness/contrast jitter
    training=False: resize the shorter side to `crop_size`, then centre crop
    """

    def __init__(
        self,
        training=True,
        crop_size=224,
        random_resize_scale=(0.3, 1.0),
        random_resize_aspect_ratio=(0.75, 1.35),
        horizontal_flip=True,
        brightness=0.0,
        contrast=0.0,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    ):
        self.training = training
        self.crop_size = crop_size
        self.scale = tuple(random_resize_scale)
        self.ratio = tuple(random_resize_aspect_ratio)
        self.horizontal_flip = horizontal_flip
        self.brightness = brightness
        self.contrast = contrast
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1, 1) * 255.0
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1, 1) * 255.0

    def __call__(self, frames):
        if isinstance(frames, np.ndarray):
            frames = torch.from_numpy(frames)
        x = frames.permute(0, 3, 1, 2).float()  # [T, C, H, W]
        T, C, H, W = x.shape
        size = self.crop_size

        if self.training:
            i, j, h, w = _random_resized_crop_params(H, W, self.scale, self.ratio)
            x = x[:, :, i : i + h, j : j + w]
            x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
            if self.horizontal_flip and torch.rand(1).item() < 0.5:
                x = x.flip(-1)
            if self.contrast > 0:
                c = _uniform(1.0 - self.contrast, 1.0 + self.contrast)
                m = x.mean()
                x = (x - m) * c + m
            if self.brightness > 0:
                x = x + _uniform(-self.brightness, self.brightness) * 255.0
            if self.contrast > 0 or self.brightness > 0:
                x = x.clamp_(0.0, 255.0)
        else:
            s = size / min(H, W)
            new_h, new_w = max(size, int(round(H * s))), max(size, int(round(W * s)))
            if (new_h, new_w) != (H, W):
                x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False, antialias=True)
            top, left = (new_h - size) // 2, (new_w - size) // 2
            x = x[:, :, top : top + size, left : left + size]

        x = x.permute(1, 0, 2, 3)  # [C, T, H, W]
        return (x - self.mean) / self.std


def build_transform(cfg_aug, training, crop_size):
    cfg_aug = cfg_aug or {}
    return VideoTransform(
        training=training,
        crop_size=crop_size,
        random_resize_scale=cfg_aug.get("random_resize_scale", (0.3, 1.0)),
        random_resize_aspect_ratio=cfg_aug.get("random_resize_aspect_ratio", (0.75, 1.35)),
        horizontal_flip=cfg_aug.get("horizontal_flip", True),
        brightness=cfg_aug.get("brightness", 0.0),
        contrast=cfg_aug.get("contrast", 0.0),
    )
