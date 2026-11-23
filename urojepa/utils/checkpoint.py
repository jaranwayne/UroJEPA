# Copyright (c) 2026 the UroJEPA authors.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from urojepa.utils.logging import get_logger

logger = get_logger(__name__)

# Keys tried, in order, when `checkpoint_key: auto`.
#   target_encoder : EMA encoder saved by `pretrain.py` (UroJEPA / UroJEPA†)
#   ema_encoder    : public V-JEPA 2.1 ViT-B/L checkpoints (vjepa2_1_vit*_dist_vitG_384.pt)
#   encoder        : fallback
AUTO_KEYS = ("target_encoder", "ema_encoder", "encoder")


def clean_state_dict(state_dict):
    out = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "").replace("backbone.", "")
        if k.startswith("encoder."):  # encoder saved inside a VideoClassifier
            k = k[len("encoder.") :]
        out[k] = v
    return out


def load_checkpoint_file(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 1.13
        return torch.load(path, map_location="cpu")


def load_encoder_weights(encoder, path, checkpoint_key="auto"):
    """Load pretrained encoder weights into a (bare) VisionTransformer.

    Works with checkpoints written by `pretrain.py` and with the public V-JEPA 2.1
    checkpoints. Parameters that only exist in V-JEPA 2.1's image tokenizer
    (`patch_embed_img`, `img_mod_embed`) are ignored, because CT videos use the
    video tokenizer only.
    """
    ckpt = load_checkpoint_file(path)
    if checkpoint_key == "auto":
        keys = [k for k in AUTO_KEYS if k in ckpt]
        if not keys:
            raise KeyError(f"None of {AUTO_KEYS} found in {path}; available keys: {list(ckpt.keys())}")
        checkpoint_key = keys[0]
    state_dict = clean_state_dict(ckpt[checkpoint_key])

    model_sd = encoder.state_dict()
    for k, v in model_sd.items():
        if k not in state_dict:
            logger.warning(f"[{path}] missing key in checkpoint: {k}")
        elif state_dict[k].shape != v.shape:
            logger.warning(f"[{path}] shape mismatch for {k}: {tuple(state_dict[k].shape)} vs {tuple(v.shape)}")
            state_dict[k] = v
    ignored = [k for k in state_dict if k not in model_sd]
    if ignored:
        logger.info(f"Ignoring {len(ignored)} checkpoint keys not used for CT video: {ignored[:6]}...")
    state_dict = {k: v for k, v in state_dict.items() if k in model_sd}
    msg = encoder.load_state_dict(state_dict, strict=False)
    logger.info(f"Loaded encoder weights from {path} [{checkpoint_key}] (epoch={ckpt.get('epoch', 'n/a')}): {msg}")
    return encoder
