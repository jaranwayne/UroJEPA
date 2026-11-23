# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2026 the UroJEPA authors.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Adapted from V-JEPA 2.1 (app/vjepa_2_1/train.py). Changes for UroJEPA: CT-video
# data pipeline (dual-scale clips, hierarchical patient sampling), optional
# initialisation from public V-JEPA 2.1 weights (UroJEPA†) and torchrun support.
# The image branch, Gram-anchoring cool-down and loss-regulation code of the
# original were removed.

"""Self-supervised pre-training of UroJEPA on unlabelled CT videos.

    # single GPU
    python pretrain.py --config configs/pretrain/urojepa_vitb16.yaml
    # 8 GPUs on one node
    torchrun --nproc_per_node=8 pretrain.py --config configs/pretrain/urojepa_dagger_vitb16.yaml
"""

import argparse
import copy
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel

from urojepa.config import add_config_args, load_config
from urojepa.data.pretrain_dataset import CTPretrainDataset, make_pretrain_loader
from urojepa.data.transforms import build_transform
from urojepa.masks.distance import compute_mask_distance
from urojepa.masks.multiblock3d import MaskCollator
from urojepa.masks.utils import apply_masks
from urojepa.models.predictor import vit_predictor
from urojepa.models.vision_transformer import build_encoder
from urojepa.models.wrappers import MultiSeqWrapper, PredictorMultiSeqWrapper
from urojepa.utils.checkpoint import load_checkpoint_file, load_encoder_weights
from urojepa.utils.distributed import init_distributed
from urojepa.utils.logging import AverageMeter, CSVLogger, get_logger, gpu_timer
from urojepa.utils.misc import make_grad_scaler, parse_dtype, set_seed, unwrap
from urojepa.utils.schedulers import CosineWDSchedule, LambdaLinearWarmupHold, WarmupCosineSchedule

logger = get_logger(__name__, force=True)
LOG_FREQ = 10


def normalize_levels(h, embed_dim, num_levels):
    """Per-level LayerNorm (no affine) of concatenated hierarchical features."""
    if num_levels == 1:
        return F.layer_norm(h, (h.size(-1),))
    chunks = [F.layer_norm(h[..., i * embed_dim : (i + 1) * embed_dim], (embed_dim,)) for i in range(num_levels)]
    return torch.cat(chunks, dim=-1)


def main(cfg):
    # ------------------------------------------------------------------ config
    folder = cfg["folder"]
    cfg_meta, cfg_data, cfg_aug = cfg["meta"], cfg["data"], cfg.get("data_aug", {})
    cfg_mask, cfg_loss, cfg_model, cfg_opt = cfg["mask"], cfg["loss"], cfg["model"], cfg["optimization"]

    seed = cfg_meta.get("seed", 0)
    dtype, mixed_precision = parse_dtype(cfg_meta.get("dtype", "bfloat16"))
    use_sdpa = cfg_meta.get("use_sdpa", True)
    save_every_freq = cfg_meta.get("save_every_freq", -1)
    resume = cfg_meta.get("resume", True)

    num_frames = cfg_data.get("num_frames", 16)
    frame_steps = tuple(cfg_data.get("frame_steps", [1, 2]))
    crop_size = cfg_data.get("crop_size", 224)
    patch_size = cfg_data.get("patch_size", 16)
    tubelet_size = cfg_data.get("tubelet_size", 2)
    batch_size = cfg_data["batch_size"]  # per GPU
    grid_size = crop_size // patch_size

    loss_exp = cfg_loss.get("loss_exp", 1.0)
    predict_all = cfg_loss.get("predict_all", True)
    weight_distance_loss = cfg_loss.get("weight_distance_loss", True)
    lambda_value = cfg_loss.get("lambda_context", 0.5)
    lambda_progressive = cfg_loss.get("lambda_progressive", True)

    num_levels = cfg_model.get("num_levels", 4)

    num_epochs = cfg_opt["epochs"]
    ipe_scale = cfg_opt.get("ipe_scale", 1.0)
    ema = cfg_opt.get("ema", [0.99925, 0.99925])

    # ------------------------------------------------------------------ setup
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    world_size, rank, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True
    set_seed(seed)

    os.makedirs(folder, exist_ok=True)
    latest_path = os.path.join(folder, "latest.pth.tar")
    if rank == 0:
        with open(os.path.join(folder, "config.yaml"), "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
    logger.info(f"rank {rank}/{world_size} on {device}")

    # ------------------------------------------------------------------ models
    dataset_fpcs = [num_frames]
    encoder = build_encoder(
        cfg_model.get("model_name", "vit_base"),
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        use_sdpa=use_sdpa,
        use_activation_checkpointing=cfg_model.get("use_activation_checkpointing", False),
        interpolate_rope=cfg_model.get("interpolate_rope", True),
        modality_embedding=cfg_model.get("modality_embedding", True),
        n_output_levels=num_levels,
    )
    embed_dim = encoder.embed_dim
    init_ckpt = cfg_model.get("init_checkpoint")
    if init_ckpt:  # UroJEPA†: continue pre-training from natural-video V-JEPA 2.1 weights
        load_encoder_weights(encoder, init_ckpt, cfg_model.get("init_checkpoint_key", "auto"))
    else:  # UroJEPA: random initialisation
        logger.info("Encoder initialised from random weights")
    encoder = MultiSeqWrapper(encoder).to(device)

    predictor = vit_predictor(
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        embed_dim=embed_dim,
        predictor_embed_dim=cfg_model.get("pred_embed_dim", 384),
        depth=cfg_model.get("pred_depth", 12),
        num_heads=cfg_model.get("pred_num_heads", 12),
        num_mask_tokens=len(cfg_mask) * len(dataset_fpcs),
        zero_init_mask_tokens=cfg_model.get("zero_init_mask_tokens", True),
        use_sdpa=use_sdpa,
        use_activation_checkpointing=cfg_model.get("use_activation_checkpointing", False),
        return_all_tokens=predict_all,
        interpolate_rope=cfg_model.get("interpolate_rope", True),
        num_levels=num_levels,
    )
    predictor = PredictorMultiSeqWrapper(predictor).to(device)
    n_enc = sum(p.numel() for p in encoder.parameters())
    n_pred = sum(p.numel() for p in predictor.parameters())
    logger.info(f"Encoder parameters: {n_enc / 1e6:.1f}M, predictor parameters: {n_pred / 1e6:.1f}M")

    if world_size > 1:
        encoder = DistributedDataParallel(encoder, device_ids=[local_rank], static_graph=True)
        predictor = DistributedDataParallel(
            predictor, device_ids=[local_rank], static_graph=False, find_unused_parameters=True
        )
    # Copy after DDP has broadcast rank-0 weights, so every rank starts from the same target.
    target_encoder = copy.deepcopy(unwrap(encoder)).eval()
    for p in target_encoder.parameters():
        p.requires_grad = False

    # Different data augmentation / mask positions on every rank
    set_seed(seed + rank)

    # ------------------------------------------------------------------ data
    mask_collator = MaskCollator(
        cfgs_mask=cfg_mask,
        dataset_fpcs=dataset_fpcs,
        crop_size=crop_size,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
    )
    dataset = CTPretrainDataset(
        cfg_data["manifest"],
        root=cfg_data.get("root"),
        num_frames=num_frames,
        frame_steps=frame_steps,
        transform=build_transform(cfg_aug, training=True, crop_size=crop_size),
    )
    ipe = cfg_opt.get("ipe")
    samples_per_epoch = cfg_data.get("samples_per_epoch")
    if ipe is not None:
        samples_per_epoch = int(ipe) * batch_size * world_size
    loader, sampler = make_pretrain_loader(
        dataset,
        batch_size=batch_size,
        collator=mask_collator,
        samples_per_epoch=samples_per_epoch,
        rank=rank,
        world_size=world_size,
        seed=seed,
        num_workers=cfg_data.get("num_workers", 8),
        pin_mem=cfg_data.get("pin_mem", True),
        persistent_workers=cfg_data.get("persistent_workers", True),
    )
    ipe = len(loader)
    total_iters = ipe * num_epochs
    logger.info(f"Batch size {batch_size}/GPU x {world_size} GPUs, {ipe} iterations/epoch, {total_iters} in total")

    # ------------------------------------------------------------------ optimisation
    def _groups(module, decay):
        return (
            p
            for n, p in module.named_parameters()
            if p.requires_grad and ((("bias" not in n) and (p.ndim != 1)) == decay)
        )

    param_groups = [
        {"params": _groups(encoder, True)},
        {"params": _groups(predictor, True)},
        {"params": _groups(encoder, False), "WD_exclude": True, "weight_decay": 0},
        {"params": _groups(predictor, False), "WD_exclude": True, "weight_decay": 0},
    ]
    optimizer = torch.optim.AdamW(
        param_groups, betas=tuple(cfg_opt.get("betas", (0.9, 0.999))), eps=float(cfg_opt.get("eps", 1e-8))
    )
    T_max = int(ipe_scale * num_epochs * ipe)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(cfg_opt["warmup"] * ipe),
        start_lr=float(cfg_opt["start_lr"]),
        ref_lr=float(cfg_opt["lr"]),
        final_lr=float(cfg_opt["final_lr"]),
        T_max=T_max,
    )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=float(cfg_opt["weight_decay"]),
        final_wd=float(cfg_opt["final_weight_decay"]),
        T_max=T_max,
    )
    scaler = make_grad_scaler() if (mixed_precision and dtype == torch.float16) else None
    momentum_scheduler = (
        ema[0] + i * (ema[1] - ema[0]) / (ipe * num_epochs * ipe_scale) for i in range(int(ipe * num_epochs) + 1)
    )

    # Context (dense) loss weight: 0 before lam_start, linear ramp, then held (V-JEPA 2.1).
    lam_start, lam_end = cfg_loss.get("context_loss_warmup_iters", [15000, 30000])
    lambda_sched = LambdaLinearWarmupHold(lambda_value, lam_start, lam_end)
    if predict_all and lambda_progressive:
        logger.info(f"Context loss weight: 0 -> {lambda_value} between iterations {lam_start} and {lam_end}")
        if lam_start >= total_iters:
            logger.info(f"Training has {total_iters} iterations, so the context loss weight stays 0 in this run")

    # ------------------------------------------------------------------ resume
    start_epoch = 0
    if resume and os.path.exists(latest_path):
        ckpt = load_checkpoint_file(latest_path)
        unwrap(encoder).load_state_dict(ckpt["encoder"])
        unwrap(predictor).load_state_dict(ckpt["predictor"])
        target_encoder.load_state_dict(ckpt["target_encoder"])
        optimizer.load_state_dict(ckpt["opt"])
        if scaler is not None and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"]
        for _ in range(start_epoch * ipe):
            scheduler.step()
            wd_scheduler.step()
            next(momentum_scheduler)
            mask_collator.step()
        logger.info(f"Resumed from {latest_path} at epoch {start_epoch}")
        del ckpt

    def save_checkpoint(epoch, path, avg_loss):
        if rank != 0:
            return
        torch.save(
            {
                "encoder": unwrap(encoder).state_dict(),
                "predictor": unwrap(predictor).state_dict(),
                "target_encoder": target_encoder.state_dict(),
                "opt": optimizer.state_dict(),
                "scaler": None if scaler is None else scaler.state_dict(),
                "epoch": epoch,
                "loss": avg_loss,
                "batch_size": batch_size,
                "world_size": world_size,
                "config": cfg,
            },
            path,
        )

    # ------------------------------------------------------------------ loss
    def loss_fn(z, h, masks_to_apply, d_weights=None):
        h = [apply_masks(hi, mi, concat=False) for hi, mi in zip(h, masks_to_apply)]
        loss, n = 0.0, 0
        for i, (zi, hi) in enumerate(zip(z, h)):
            for j, (zij, hij) in enumerate(zip(zi, hi)):
                l_ij = torch.abs(zij - hij) ** loss_exp
                if d_weights is not None:
                    l_ij = l_ij / d_weights[i][j].unsqueeze(2)
                loss = loss + torch.mean(l_ij) / loss_exp
                n += 1
        return loss / n

    if rank == 0:
        csv_logger = CSVLogger(
            os.path.join(folder, "pretrain_log.csv"),
            ("%d", "epoch"),
            ("%d", "itr"),
            ("%.5f", "loss"),
            ("%.5f", "loss_pred"),
            ("%.5f", "loss_context"),
            ("%.4f", "lambda_context"),
            ("%.3e", "lr"),
            ("%d", "iter_ms"),
            ("%d", "gpu_ms"),
            ("%d", "data_ms"),
        )
        epoch_logger = CSVLogger(os.path.join(folder, "pretrain_epoch_loss.csv"), ("%d", "epoch"), ("%.5f", "loss"))

    # ------------------------------------------------------------------ training loop
    for epoch in range(start_epoch, num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        sampler.set_epoch(epoch)
        loss_meter, time_meter, gpu_meter, data_meter = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        encoder.train()
        predictor.train()

        itr_start = time.time()
        for itr, sample in enumerate(loader):
            clips = [s[0].to(device, non_blocking=True) for s in sample]
            masks_enc = [[m.to(device, non_blocking=True) for m in s[1]] for s in sample]
            masks_pred = [[m.to(device, non_blocking=True) for m in s[2]] for s in sample]
            data_ms = (time.time() - itr_start) * 1000.0
            global_itr = epoch * ipe + itr

            def train_step():
                new_lr = scheduler.step()
                scheduler_wd = wd_scheduler.step()
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=mixed_precision):
                    with torch.no_grad():
                        h = target_encoder(clips, training_mode=True)
                        h = [normalize_levels(hi, embed_dim, num_levels) for hi in h]
                    z = encoder(clips, masks_enc, training_mode=True)
                    z_pred, z_context = predictor(z, masks_enc, masks_pred)

                    loss_pred = loss_fn(z_pred, h, masks_pred)
                    loss = loss_pred
                    loss_ctx = torch.zeros((), device=device)
                    lam = 0.0
                    if predict_all:
                        lam = lambda_sched.value(global_itr) if lambda_progressive else lambda_value
                        d_weights = (
                            compute_mask_distance(masks_pred, masks_enc, grid_size) if weight_distance_loss else None
                        )
                        loss_ctx = loss_fn(z_context, h, masks_enc, d_weights=d_weights)
                        loss = loss + lam * loss_ctx

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                # EMA update of the target encoder
                m = min(next(momentum_scheduler), ema[1])
                with torch.no_grad():
                    params_q = list(unwrap(encoder).parameters())
                    params_k = list(target_encoder.parameters())
                    torch._foreach_mul_(params_k, m)
                    torch._foreach_add_(params_k, params_q, alpha=1 - m)
                return float(loss), float(loss_pred), float(loss_ctx), lam, new_lr, scheduler_wd

            (loss, loss_pred, loss_ctx, lam, lr, _), gpu_ms = gpu_timer(train_step)
            iter_ms = (time.time() - itr_start) * 1000.0
            loss_meter.update(loss)
            time_meter.update(iter_ms)
            gpu_meter.update(gpu_ms)
            data_meter.update(data_ms)
            if not np.isfinite(loss):
                raise RuntimeError(f"Loss is {loss} at epoch {epoch + 1}, iteration {itr}")

            if rank == 0:
                csv_logger.log(epoch + 1, itr, loss, loss_pred, loss_ctx, lam, lr, iter_ms, gpu_ms, data_ms)
                if itr % LOG_FREQ == 0 or itr == ipe - 1:
                    mem = torch.cuda.max_memory_allocated() / 1024.0**2 if device.type == "cuda" else 0
                    logger.info(
                        f"[{epoch + 1}, {itr:5d}] loss {loss_meter.avg:.4f} (pred {loss_pred:.4f}, ctx {loss_ctx:.4f}, "
                        f"lambda {lam:.3f}) lr {lr:.2e} mem {mem:.0f}MB iter {time_meter.avg:.0f}ms "
                        f"gpu {gpu_meter.avg:.0f}ms data {data_meter.avg:.0f}ms"
                    )
            itr_start = time.time()

        if world_size > 1:
            t = torch.tensor([loss_meter.avg], device=device)
            dist.all_reduce(t)
            epoch_loss = float(t.item() / world_size)
        else:
            epoch_loss = loss_meter.avg
        logger.info(f"Epoch {epoch + 1}: average loss {epoch_loss:.4f}")
        if rank == 0:
            epoch_logger.log(epoch + 1, epoch_loss)
        save_checkpoint(epoch + 1, latest_path, epoch_loss)
        if save_every_freq > 0 and (epoch + 1) % save_every_freq == 0:
            save_checkpoint(epoch + 1, os.path.join(folder, f"e{epoch + 1}.pth.tar"), epoch_loss)

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = add_config_args(argparse.ArgumentParser(description=__doc__.split("\n\n")[0]))
    args = parser.parse_args()
    main(load_config(args.config, args.opts))
