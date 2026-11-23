# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Adapted from V-JEPA 2.1 (app/vjepa_2_1/models/vision_transformer.py).
# Changes for UroJEPA: video-only tokenizer (the image tokenizer of V-JEPA 2.1 is
# not used for CT videos), 3D-RoPE only, configurable number of hierarchical
# output levels. Parameter names are unchanged, so the public V-JEPA 2.1
# checkpoints load directly (used to initialise UroJEPA†).

import math
from functools import partial

import torch
import torch.nn as nn
import torch.utils.checkpoint

from urojepa.masks.utils import apply_masks
from urojepa.models.modules import Block
from urojepa.utils.tensors import trunc_normal_

# Encoder layers whose (normalised) outputs are concatenated as JEPA targets
# ("deep self-supervision" in V-JEPA 2.1).
HIERARCHICAL_LAYERS = {
    12: [2, 5, 8, 11],
    24: [5, 11, 17, 23],
    40: [9, 19, 29, 39],
    48: [11, 23, 37, 47],
}


class PatchEmbed3D(nn.Module):
    """Video (tubelet) to patch embedding."""

    def __init__(self, patch_size=16, tubelet_size=2, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class VisionTransformer(nn.Module):
    """Video Vision Transformer encoder (V-JEPA 2.1 style)."""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        num_frames=16,
        tubelet_size=2,
        in_chans=3,
        embed_dim=768,
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
        use_sdpa=True,
        use_activation_checkpointing=False,
        is_causal=False,
        interpolate_rope=True,
        modality_embedding=True,
        n_output_levels=4,
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.img_height, self.img_width = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.use_activation_checkpointing = use_activation_checkpointing

        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size, tubelet_size=tubelet_size, in_chans=in_chans, embed_dim=embed_dim
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
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
                    grid_size=img_size[0] // patch_size,
                    interpolate_rope=interpolate_rope,
                    patch_size=patch_size,
                )
                for i in range(depth)
            ]
        )

        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()

        if depth in HIERARCHICAL_LAYERS:
            self.hierarchical_layers = HIERARCHICAL_LAYERS[depth]
        else:
            q = depth // 4
            self.hierarchical_layers = [q - 1, 2 * q - 1, 3 * q - 1, depth - 1]
        assert 1 <= n_output_levels <= len(self.hierarchical_layers)
        self.out_layers_distillation = self.hierarchical_layers[-n_output_levels:]
        self.norms_block = nn.ModuleList([norm_layer(embed_dim) for _ in self.hierarchical_layers])

        # Learned "video" modality embedding of V-JEPA 2.1 (kept for weight compatibility).
        self.modality_embedding = modality_embedding
        if modality_embedding:
            self.video_mod_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.normal_(self.video_mod_embed, std=1e-6)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _rescale_blocks(self):
        for layer_id, layer in enumerate(self.blocks):
            layer.attn.proj.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))
            layer.mlp.fc2.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))

    def get_num_layers(self):
        return len(self.blocks)

    def forward(self, x, masks=None, training=False):
        """
        :param x: video clip, [B, C, T, H, W]
        :param masks: (list of) [B, K] indices of tokens to keep (context tokens)
        :param training: if True, return the concatenation of the normalised hierarchical
            levels [B, N, n_levels * D] (JEPA targets / predictor input); otherwise return
            the normalised last-layer tokens [B, N, D] (used for fine-tuning).
        """
        if masks is not None and not isinstance(masks, list):
            masks = [masks]

        _, _, T, H, W = x.shape
        T = T // self.tubelet_size
        H_patches = H // self.patch_size
        W_patches = W // self.patch_size

        x = self.patch_embed(x)
        if self.modality_embedding:
            x = x + self.video_mod_embed

        if masks is not None:
            x = apply_masks(x, masks)
            masks = torch.cat(masks, dim=0)

        hier = []
        for i, blk in enumerate(self.blocks):
            if self.use_activation_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    blk, x, masks, T, H_patches, W_patches, use_reentrant=False
                )
            else:
                x = blk(x, mask=masks, T=T, H_patches=H_patches, W_patches=W_patches)
            if training and i in self.out_layers_distillation:
                out_idx = self.hierarchical_layers.index(i)
                hier.append(self.norms_block[out_idx](x))

        if training:
            return torch.cat(hier, dim=2)
        return self.norms_block[-1](x)


def vit_small(patch_size=16, **kwargs):
    return VisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def vit_base(patch_size=16, **kwargs):
    return VisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def vit_large(patch_size=16, **kwargs):
    return VisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


MODELS = {"vit_small": vit_small, "vit_base": vit_base, "vit_large": vit_large}


def build_encoder(model_name="vit_base", **kwargs):
    if model_name not in MODELS:
        raise ValueError(f"Unknown model_name {model_name!r}; choose from {sorted(MODELS)}")
    return MODELS[model_name](**kwargs)
