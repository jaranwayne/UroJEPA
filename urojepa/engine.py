# Copyright (c) 2026 the UroJEPA authors.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Model construction and video-level inference shared by finetune.py and evaluate.py."""

import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from urojepa.data.finetune_dataset import CTClassificationDataset, make_finetune_loader
from urojepa.data.transforms import build_transform
from urojepa.models.classifier import VideoClassifier
from urojepa.models.vision_transformer import build_encoder
from urojepa.utils.checkpoint import load_encoder_weights
from urojepa.utils.distributed import all_gather_object
from urojepa.utils.logging import get_logger
from urojepa.utils.metrics import binary_metrics

logger = get_logger(__name__)


def build_classifier(cfg, load_pretrained=True):
    cfg_model, cfg_data = cfg["model"], cfg["data"]
    encoder = build_encoder(
        cfg_model.get("model_name", "vit_base"),
        img_size=cfg_data.get("crop_size", 224),
        patch_size=cfg_model.get("patch_size", 16),
        num_frames=cfg_data.get("num_frames", 16),
        tubelet_size=cfg_model.get("tubelet_size", 2),
        use_sdpa=cfg_model.get("use_sdpa", True),
        use_activation_checkpointing=cfg_model.get("use_activation_checkpointing", False),
        interpolate_rope=cfg_model.get("interpolate_rope", True),
        modality_embedding=cfg_model.get("modality_embedding", True),
        drop_path_rate=cfg_model.get("drop_path", 0.0),
    )
    pretrained = cfg_model.get("pretrained")
    if load_pretrained:
        if pretrained:
            load_encoder_weights(encoder, pretrained, cfg_model.get("checkpoint_key", "auto"))
        else:
            logger.warning("model.pretrained is empty: the encoder is trained from random initialisation")
    return VideoClassifier(
        encoder,
        num_classes=cfg_model.get("num_classes", 2),
        head=cfg_model.get("head", "attentive"),
        head_depth=cfg_model.get("head_depth", 4),
        head_num_heads=cfg_model.get("head_num_heads", 16),
    )


def make_eval_loader(df, cfg, rank=0, world_size=1):
    cfg_data = cfg["data"]
    dataset = CTClassificationDataset(
        df,
        training=False,
        transform=build_transform(cfg.get("data_aug"), training=False, crop_size=cfg_data.get("crop_size", 224)),
        num_frames=cfg_data.get("num_frames", 16),
        frame_steps=cfg_data.get("frame_steps", [1, 2]),
        eval_num_clips=cfg_data.get("eval_num_clips", 9),
        eval_frame_step=cfg_data.get("eval_frame_step", 1),
    )
    loader, _ = make_finetune_loader(
        dataset,
        batch_size=cfg_data.get("eval_batch_size", 4),
        training=False,
        rank=rank,
        world_size=world_size,
        num_workers=cfg_data.get("num_workers", 8),
    )
    return loader


@torch.no_grad()
def predict(model, loader, device, dtype=torch.bfloat16, mixed_precision=True):
    """Video-level P(positive): logits are averaged over the K clips of each video,
    then passed through a softmax. Returns (indices, labels, probs) sorted by index,
    gathered from all ranks without the padding duplicates of DistributedSampler."""
    model.eval()
    records = []
    for clips, labels, index in loader:
        B, K = clips.shape[:2]
        x = clips.flatten(0, 1).to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=mixed_precision):
            logits = model(x)
        logits = logits.float().view(B, K, -1).mean(dim=1)
        probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        records += list(zip(index.tolist(), labels.tolist(), probs.tolist()))

    merged = {}
    for part in all_gather_object(records):
        for i, y, p in part:
            merged[int(i)] = (int(y), float(p))
    idx = np.array(sorted(merged))
    labels = np.array([merged[i][0] for i in idx])
    probs = np.array([merged[i][1] for i in idx])
    return idx, labels, probs


def log_loss(labels, probs, eps=1e-7):
    p = np.clip(probs, eps, 1 - eps)
    return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))


def evaluate_and_save(model, df, name, cfg, out_dir, device, dtype, mixed_precision, rank, world_size, threshold=0.5):
    """Predict on `df`, write `predictions_<name>.csv` and return the metrics (rank 0 writes)."""
    df = df.reset_index(drop=True)
    loader = make_eval_loader(df, cfg, rank=rank, world_size=world_size)
    idx, labels, probs = predict(model, loader, device, dtype, mixed_precision)
    metrics = binary_metrics(labels, probs, threshold)
    metrics["log_loss"] = log_loss(labels, probs)

    out = df.iloc[idx].copy()
    out["prob"] = probs
    out["pred"] = (probs >= threshold).astype(int)
    if "center" in out.columns:
        metrics["per_center"] = {
            str(c): binary_metrics(g["label"].values, g["prob"].values, threshold) for c, g in out.groupby("center")
        }
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        out.to_csv(os.path.join(out_dir, f"predictions_{name}.csv"), index=False)
    logger.info(
        f"[{name}] n={metrics['n']} (pos {metrics['n_pos']}) AUROC {metrics['auroc']:.4f} "
        f"acc {metrics['accuracy']:.4f} sens {metrics['sensitivity']:.4f} spec {metrics['specificity']:.4f}"
    )
    return metrics


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
