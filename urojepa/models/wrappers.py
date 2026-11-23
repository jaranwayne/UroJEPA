# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Adapted from V-JEPA 2.1 (app/vjepa_2_1/wrappers.py); the Gram-anchoring branch
# (cool-down only) was removed.

import torch.nn as nn


class MultiSeqWrapper(nn.Module):
    """Runs the encoder on a list of clip batches (one per clip length) and,
    optionally, on several masks per batch."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = backbone.embed_dim

    def forward(self, x, masks=None, training_mode=False):
        """
        :param x: list of [B, C, T, H, W] tensors (one per clip length)
        :param masks: list (per clip length) of lists (per mask generator) of [B, K] index tensors
        """
        if masks is None:
            return [self.backbone(x_fpc, training=training_mode) for x_fpc in x]

        outs = [[] for _ in x]
        for i, (x_fpc, m_fpc) in enumerate(zip(x, masks)):
            for m in m_fpc:
                outs[i] += [self.backbone(x_fpc, masks=m, training=training_mode)]
        return outs


class PredictorMultiSeqWrapper(nn.Module):

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x, masks_x, masks_y):
        outs_pred = [[] for _ in x]
        outs_context = [[] for _ in x]
        for i, (x_fpc, mx_fpc, my_fpc) in enumerate(zip(x, masks_x, masks_y)):
            for xij, mx, my in zip(x_fpc, mx_fpc, my_fpc):
                x_pred, x_context = self.backbone(xij, mx, my, mask_index=i)
                outs_pred[i] += [x_pred]
                outs_context[i] += [x_context]
        return outs_pred, outs_context
