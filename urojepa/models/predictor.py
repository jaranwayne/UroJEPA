# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Adapted from V-JEPA 2.1 (app/vjepa_2_1/models/predictor.py).
# Changes for UroJEPA: 3D-RoPE only, video-only, no register/CLS tokens; the number
# of hierarchical target levels is an explicit argument; token re-ordering uses
# torch.gather (numerically identical to the original loop).

import math
from functools import partial

import torch
import torch.nn as nn
import torch.utils.checkpoint

from urojepa.masks.utils import apply_masks
from urojepa.models.modules import Block
from urojepa.utils.tensors import trunc_normal_


def _gather_tokens(x, index):
    return torch.gather(x, 1, index.unsqueeze(-1).expand(-1, -1, x.size(-1)))


class VisionTransformerPredictor(nn.Module):
    """JEPA predictor: predicts target-encoder features of masked tokens (and, with
    `return_all_tokens=True`, of the visible context tokens: V-JEPA 2.1 dense loss)."""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        num_frames=16,
        tubelet_size=2,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        num_mask_tokens=2,
        zero_init_mask_tokens=True,
        use_sdpa=True,
        is_causal=False,
        use_activation_checkpointing=False,
        return_all_tokens=True,
        interpolate_rope=True,
        num_levels=4,
    ):
        super().__init__()
        self.return_all_tokens = return_all_tokens
        self.num_levels = num_levels
        self.use_activation_checkpointing = use_activation_checkpointing

        # Input: concatenation of `num_levels` encoder levels.
        if num_levels == 1:
            self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        else:
            self.predictor_embed = nn.Sequential(
                nn.Linear(embed_dim * num_levels, embed_dim, bias=True),
                nn.GELU(),
                nn.Linear(embed_dim, predictor_embed_dim, bias=True),
            )

        self.num_mask_tokens = num_mask_tokens
        self.mask_tokens = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, 1, predictor_embed_dim)) for _ in range(num_mask_tokens)]
        )

        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.grid_height = img_size[0] // patch_size
        self.grid_width = img_size[1] // patch_size
        self.grid_depth = num_frames // tubelet_size
        self.num_patches = self.grid_depth * self.grid_height * self.grid_width

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.predictor_blocks = nn.ModuleList(
            [
                Block(
                    dim=predictor_embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    use_sdpa=use_sdpa,
                    is_causal=is_causal,
                    grid_size=self.grid_height,
                    interpolate_rope=interpolate_rope,
                    patch_size=patch_size,
                )
                for i in range(depth)
            ]
        )

        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, num_levels * embed_dim, bias=True)
        if self.return_all_tokens:
            self.predictor_proj_context = nn.Linear(predictor_embed_dim, num_levels * embed_dim, bias=True)

        self.init_std = init_std
        if not zero_init_mask_tokens:
            for mt in self.mask_tokens:
                trunc_normal_(mt, std=init_std)
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _rescale_blocks(self):
        for layer_id, layer in enumerate(self.predictor_blocks):
            layer.attn.proj.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))
            layer.mlp.fc2.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))

    def forward(self, x, masks_x, masks_y, mask_index=0):
        """
        :param x: context tokens from the encoder, [B * len(masks_x), N_ctxt, num_levels * D]
        :param masks_x: (list of) [B, N_ctxt] indices of context tokens
        :param masks_y: (list of) [B, N_pred] indices of target (masked) tokens
        :returns: (pred, context) features with num_levels * D channels
        """
        if not isinstance(masks_x, list):
            masks_x = [masks_x]
        if not isinstance(masks_y, list):
            masks_y = [masks_y]
        B = len(x) // len(masks_x)

        x = self.predictor_embed(x)
        _, N_ctxt, _ = x.shape

        pred_tokens = self.mask_tokens[mask_index % self.num_mask_tokens]
        pred_tokens = pred_tokens.repeat(B, self.num_patches, 1)
        pred_tokens = apply_masks(pred_tokens, masks_y)

        x = x.repeat(len(masks_x), 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)

        masks_x = torch.cat(masks_x, dim=0)
        masks_y = torch.cat(masks_y, dim=0)
        masks = torch.cat([masks_x, masks_y], dim=1)

        # Sort tokens by their position so that RoPE sees the true 3D location.
        argsort = torch.argsort(masks, dim=1)
        masks = torch.gather(masks, 1, argsort)
        x = _gather_tokens(x, argsort)

        for blk in self.predictor_blocks:
            if self.use_activation_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, masks, use_reentrant=False)
            else:
                x = blk(x, mask=masks)
        x = self.predictor_norm(x)

        reverse_argsort = torch.argsort(argsort, dim=1)
        x = _gather_tokens(x, reverse_argsort)
        x_pred = self.predictor_proj(x[:, N_ctxt:, :])
        if not self.return_all_tokens:
            return x_pred, None
        x_context = self.predictor_proj_context(x[:, :N_ctxt, :])
        return x_pred, x_context


def vit_predictor(**kwargs):
    return VisionTransformerPredictor(
        mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
