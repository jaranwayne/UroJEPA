# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Adapted from V-JEPA 2.1 (app/vjepa_2_1/models/utils/masks_dist.py).
# The per-token loop of the original was vectorised with a single torch.cdist
# call; the returned weights are identical.

import torch


def token_positions(ids, grid_size):
    """Map flattened token indices to (depth, height, width) grid coordinates."""
    tokens_per_frame = grid_size * grid_size
    d = torch.div(ids, tokens_per_frame, rounding_mode="floor")
    rem = ids - d * tokens_per_frame
    h = torch.div(rem, grid_size, rounding_mode="floor")
    w = rem - h * grid_size
    return torch.stack([d, h, w], dim=-1).float()


@torch.no_grad()
def compute_mask_distance(masks_pred, masks_enc, grid_size):
    """For every context token, sqrt of the Euclidean distance (in token units) to the
    nearest masked token. Used to down-weight the V-JEPA 2.1 context loss for context
    tokens far from any masked region.

    :param masks_pred: [fpc][mask] of [B, N_pred] index tensors
    :param masks_enc:  [fpc][mask] of [B, N_enc] index tensors
    :returns: [fpc][mask] of [B, N_enc] distance tensors
    """
    distances = []
    for masks_pred_i, masks_enc_i in zip(masks_pred, masks_enc):
        row = []
        for mp, me in zip(masks_pred_i, masks_enc_i):
            pos_enc = token_positions(me, grid_size)  # [B, N_enc, 3]
            pos_pred = token_positions(mp, grid_size)  # [B, N_pred, 3]
            dmin = torch.cdist(pos_enc, pos_pred, p=2).min(dim=-1).values  # [B, N_enc]
            row.append(dmin.sqrt())
        distances.append(row)
    return distances
