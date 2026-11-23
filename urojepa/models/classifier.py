# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2026 the UroJEPA authors.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# AttentivePooler / AttentiveClassifier are adapted from V-JEPA 2
# (src/models/attentive_pooler.py).

import math

import torch
import torch.nn as nn

from urojepa.models.modules import CrossAttentionBlock, SelfAttentionBlock
from urojepa.utils.tensors import trunc_normal_


class AttentivePooler(nn.Module):
    """(depth - 1) self-attention blocks followed by cross-attention pooling with learnable queries."""

    def __init__(
        self,
        num_queries=1,
        embed_dim=768,
        num_heads=12,
        mlp_ratio=4.0,
        depth=1,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        qkv_bias=True,
    ):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, embed_dim))
        self.cross_attention_block = CrossAttentionBlock(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer
        )
        self.blocks = None
        if depth > 1:
            self.blocks = nn.ModuleList(
                [
                    SelfAttentionBlock(
                        dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer
                    )
                    for _ in range(depth - 1)
                ]
            )
        self.init_std = init_std
        trunc_normal_(self.query_tokens, std=self.init_std)
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _rescale_blocks(self):
        layer_id = 0
        if self.blocks is not None:
            for layer_id, layer in enumerate(self.blocks):
                layer.attn.proj.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))
                layer.mlp.fc2.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))
        self.cross_attention_block.mlp.fc2.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        if self.blocks is not None:
            for blk in self.blocks:
                x = blk(x)
        q = self.query_tokens.repeat(len(x), 1, 1)
        return self.cross_attention_block(q, x)


class AttentiveClassifier(nn.Module):
    """Attentive probe of V-JEPA 2 / 2.1: attentive pooling followed by a linear layer."""

    def __init__(self, embed_dim=768, num_heads=16, depth=4, mlp_ratio=4.0, num_classes=2, init_std=0.02):
        super().__init__()
        self.pooler = AttentivePooler(
            num_queries=1,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            depth=depth,
            init_std=init_std,
            qkv_bias=True,
        )
        self.linear = nn.Linear(embed_dim, num_classes, bias=True)

    def forward(self, x):
        return self.linear(self.pooler(x).squeeze(1))


class MeanPoolClassifier(nn.Module):
    """Mean pooling over all tokens -> LayerNorm -> linear."""

    def __init__(self, embed_dim=768, num_classes=2, init_std=0.02):
        super().__init__()
        self.fc_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.linear = nn.Linear(embed_dim, num_classes)
        trunc_normal_(self.linear.weight, std=init_std)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        return self.linear(self.fc_norm(x.mean(dim=1)))


class VideoClassifier(nn.Module):
    """Encoder + classification head, fine-tuned end to end.

    Input: clips [B, C, T, H, W]; output: logits [B, num_classes].
    """

    def __init__(self, encoder, num_classes=2, head="attentive", head_depth=4, head_num_heads=16):
        super().__init__()
        self.encoder = encoder
        embed_dim = encoder.embed_dim
        if head == "attentive":
            self.head = AttentiveClassifier(
                embed_dim, num_heads=head_num_heads, depth=head_depth, num_classes=num_classes
            )
        elif head == "mean_linear":
            self.head = MeanPoolClassifier(embed_dim, num_classes=num_classes)
        else:
            raise ValueError(f"Unknown head {head!r} (use 'attentive' or 'mean_linear')")

        # Only the last hierarchical LayerNorm of the encoder is used for fine-tuning;
        # freeze the others so that DistributedDataParallel sees no unused parameters.
        for norm in encoder.norms_block[:-1]:
            for p in norm.parameters():
                p.requires_grad = False

    def forward(self, x):
        return self.head(self.encoder(x, training=False))


def param_groups(model, weight_decay=0.01):
    """AdamW parameter groups: no weight decay for biases, norms, embeddings and query tokens."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or name.endswith(".bias") or "mod_embed" in name or "query_tokens" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
