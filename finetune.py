# Copyright (c) 2026 the UroJEPA authors.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""End-to-end fine-tuning of a pretrained encoder for one binary outcome.

The encoder and a classification head are trained on the training split; the
checkpoint with the best validation AUROC is kept and evaluated on the internal test
split and on any external cohorts (video-level, logits averaged over 9 clips/video).

    python finetune.py --config configs/finetune/tasks/stone_sepsis.yaml \
        --opts seed=0 model.pretrained=outputs/pretrain/urojepa_dagger_vitb16/latest.pth.tar

    # label-efficiency experiment (25 % of training patients)
    python finetune.py --config configs/finetune/tasks/stone_sepsis.yaml --opts seed=0 data.train_fraction=0.25
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
from torch.nn.parallel import DistributedDataParallel

from urojepa.config import add_config_args, load_config
from urojepa.data.finetune_dataset import CTClassificationDataset, make_finetune_loader
from urojepa.data.manifest import read_manifest, subsample_patients
from urojepa.data.transforms import build_transform
from urojepa.engine import build_classifier, evaluate_and_save, log_loss, make_eval_loader, predict, save_json
from urojepa.models.classifier import param_groups
from urojepa.utils.checkpoint import load_checkpoint_file
from urojepa.utils.distributed import all_reduce_mean, barrier, init_distributed
from urojepa.utils.logging import AverageMeter, CSVLogger, get_logger
from urojepa.utils.metrics import binary_metrics
from urojepa.utils.misc import make_grad_scaler, parse_dtype, set_seed, unwrap
from urojepa.utils.schedulers import WarmupCosineSchedule

logger = get_logger(__name__, force=True)


def output_dir(cfg):
    frac = cfg["data"].get("train_fraction", 1.0)
    parts = [cfg["folder"]]
    if frac is not None and frac < 1.0:
        parts.append(f"frac{frac:g}")
    parts.append(f"seed{cfg['seed']}")
    return os.path.join(*parts)


def main(cfg):
    seed = cfg.get("seed", 0)
    cfg["seed"] = seed
    cfg_data, cfg_opt = cfg["data"], cfg["optimization"]
    dtype, mixed_precision = parse_dtype(cfg_opt.get("dtype", "bfloat16"))
    select_metric = cfg_opt.get("select_metric", "auroc")
    threshold = cfg.get("threshold", 0.5)

    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    world_size, rank, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True

    out_dir = output_dir(cfg)
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
    logger.info(f"Output directory: {out_dir}")

    # ------------------------------------------------------------------ data
    root = cfg_data.get("root")
    train_df = read_manifest(cfg_data["train"], root=root, require_label=True)
    val_df = read_manifest(cfg_data["val"], root=root, require_label=True)
    frac = cfg_data.get("train_fraction", 1.0)
    if frac is not None and frac < 1.0:
        frac_seed = cfg_data.get("train_fraction_seed")
        train_df = subsample_patients(train_df, frac, seed if frac_seed is None else frac_seed)
    logger.info(
        f"Train: {len(train_df)} videos / {train_df.patient_id.nunique()} patients "
        f"({int(train_df.label.sum())} positive videos); val: {len(val_df)} videos"
    )

    total_bs = cfg_opt.get("batch_size", 16)
    if total_bs % world_size != 0:
        raise ValueError(f"batch_size {total_bs} must be divisible by the number of GPUs ({world_size})")
    batch_size = total_bs // world_size

    set_seed(seed)
    train_set = CTClassificationDataset(
        train_df,
        training=True,
        transform=build_transform(cfg.get("data_aug"), training=True, crop_size=cfg_data.get("crop_size", 224)),
        num_frames=cfg_data.get("num_frames", 16),
        frame_steps=cfg_data.get("frame_steps", [1, 2]),
    )
    train_loader, train_sampler = make_finetune_loader(
        train_set,
        batch_size=batch_size,
        training=True,
        rank=rank,
        world_size=world_size,
        seed=seed,
        num_workers=cfg_data.get("num_workers", 8),
    )
    val_loader = make_eval_loader(val_df, cfg, rank=rank, world_size=world_size)

    # ------------------------------------------------------------------ model
    model = build_classifier(cfg).to(device)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank])
    set_seed(seed + rank)  # different augmentations on every rank

    num_epochs = cfg_opt.get("epochs", 30)
    ipe = len(train_loader)
    optimizer = torch.optim.AdamW(
        param_groups(unwrap(model), weight_decay=float(cfg_opt.get("weight_decay", 0.01))),
        betas=tuple(cfg_opt.get("betas", (0.9, 0.999))),
        eps=float(cfg_opt.get("eps", 1e-8)),
    )
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(cfg_opt.get("warmup_epochs", 5) * ipe),
        start_lr=float(cfg_opt.get("start_lr", 0.0)),
        ref_lr=float(cfg_opt.get("lr", 1e-3)),
        final_lr=float(cfg_opt.get("final_lr", 0.0)),
        T_max=num_epochs * ipe,
    )
    scaler = make_grad_scaler() if (mixed_precision and dtype == torch.float16) else None
    criterion = torch.nn.CrossEntropyLoss()

    if rank == 0:
        csv_logger = CSVLogger(
            os.path.join(out_dir, "log.csv"),
            ("%d", "epoch"),
            ("%.5f", "train_loss"),
            ("%.5f", "val_loss"),
            ("%.5f", "val_auroc"),
            ("%.5f", "val_accuracy"),
            ("%.3e", "lr"),
        )
    best_path = os.path.join(out_dir, "best.pt")
    best_score, best_epoch, best_val = None, -1, None

    # ------------------------------------------------------------------ training
    for epoch in range(num_epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        loss_meter = AverageMeter()
        t0 = time.time()
        for itr, (clips, labels, _) in enumerate(train_loader):
            lr = scheduler.step()
            clips = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=mixed_precision):
                logits = model(clips)
            loss = criterion(logits.float(), labels)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if not np.isfinite(loss.item()):
                raise RuntimeError(f"Loss is {loss.item()} at epoch {epoch + 1}")
            loss_meter.update(loss.item(), len(labels))
            if itr % 10 == 0:
                logger.info(f"[{epoch + 1}, {itr:4d}/{ipe}] loss {loss_meter.avg:.4f} lr {lr:.2e}")
        train_loss = all_reduce_mean(loss_meter.avg)

        # -- validation (checkpoint selection)
        _, y_val, p_val = predict(model, val_loader, device, dtype, mixed_precision)
        val = binary_metrics(y_val, p_val, threshold)
        val["log_loss"] = log_loss(y_val, p_val)
        score = {"auroc": val["auroc"], "accuracy": val["accuracy"], "loss": -val["log_loss"]}[select_metric]
        score = score if np.isfinite(score) else -np.inf
        improved = best_score is None or score > best_score
        logger.info(
            f"Epoch {epoch + 1}/{num_epochs} ({time.time() - t0:.0f}s): train loss {train_loss:.4f} | "
            f"val loss {val['log_loss']:.4f} AUROC {val['auroc']:.4f} acc {val['accuracy']:.4f}"
            + (" *" if improved else "")
        )
        if rank == 0:
            csv_logger.log(epoch + 1, train_loss, val["log_loss"], val["auroc"], val["accuracy"], lr)
        if improved:
            best_score, best_epoch, best_val = score, epoch + 1, val
            if rank == 0:
                torch.save(
                    {"model": unwrap(model).state_dict(), "config": cfg, "epoch": epoch + 1, "val_metrics": val},
                    best_path,
                )
        barrier()

    # ------------------------------------------------------------------ final evaluation
    logger.info(f"Best validation {select_metric} at epoch {best_epoch}; evaluating {best_path}")
    unwrap(model).load_state_dict(load_checkpoint_file(best_path)["model"])
    results = {"best_epoch": best_epoch, "select_metric": select_metric, "val": best_val}
    eval_sets = {}
    if cfg_data.get("test"):
        eval_sets["test"] = cfg_data["test"]
    for name, path in (cfg_data.get("external") or {}).items():
        if path and os.path.exists(path):
            eval_sets[name] = path
        else:
            logger.warning(f"External cohort {name!r} not found at {path}; skipped")
    for name, path in eval_sets.items():
        df = read_manifest(path, root=root, require_label=True)
        results[name] = evaluate_and_save(
            model, df, name, cfg, out_dir, device, dtype, mixed_precision, rank, world_size, threshold
        )
    if rank == 0:
        save_json(results, os.path.join(out_dir, "metrics.json"))
        logger.info(f"Saved metrics to {os.path.join(out_dir, 'metrics.json')}")

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = add_config_args(argparse.ArgumentParser(description=__doc__.split("\n\n")[0]))
    args = parser.parse_args()
    main(load_config(args.config, args.opts))
