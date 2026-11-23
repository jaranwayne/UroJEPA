# Copyright (c) 2026 the UroJEPA authors.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Apply a fine-tuned UroJEPA model (best.pt written by finetune.py) to new CT videos,
e.g. external validation cohorts or the reader-study cohort.

    python evaluate.py --checkpoint outputs/finetune/stone_sepsis/urojepa_dagger/seed0/best.pt \
        --csv external=data/stone_sepsis/external.csv reader=data/stone_sepsis/reader_study.csv \
        --out outputs/eval/stone_sepsis/seed0

Each CSV needs `patient_id,video_path,label` (`label` may be all -1 when unknown;
metrics are then skipped). Writes predictions_<name>.csv and metrics.json.
"""

import argparse
import os

import torch
import torch.multiprocessing as mp

from urojepa.config import apply_overrides
from urojepa.data.manifest import read_manifest
from urojepa.engine import build_classifier, evaluate_and_save, make_eval_loader, predict, save_json
from urojepa.utils.checkpoint import load_checkpoint_file
from urojepa.utils.distributed import init_distributed
from urojepa.utils.logging import get_logger
from urojepa.utils.misc import parse_dtype

logger = get_logger(__name__, force=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", required=True, help="best.pt from finetune.py")
    parser.add_argument("--csv", nargs="+", required=True, help="name=path.csv (or just path.csv)")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--root", default=None, help="prefix for relative video paths")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--opts", nargs="*", default=[], help="config overrides, e.g. data.eval_batch_size=8")
    args = parser.parse_args()

    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    world_size, rank, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")

    ckpt = load_checkpoint_file(args.checkpoint)
    cfg = apply_overrides(ckpt["config"], args.opts)
    dtype, mixed_precision = parse_dtype(cfg["optimization"].get("dtype", "bfloat16"))

    model = build_classifier(cfg, load_pretrained=False)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    logger.info(f"Loaded {args.checkpoint} (epoch {ckpt.get('epoch')})")

    results = {"checkpoint": os.path.abspath(args.checkpoint)}
    for item in args.csv:
        name, path = item.split("=", 1) if "=" in item else (os.path.splitext(os.path.basename(item))[0], item)
        df = read_manifest(path, root=args.root)
        if "label" in df.columns and (df["label"] >= 0).all():
            df["label"] = df["label"].astype(int)
            results[name] = evaluate_and_save(
                model, df, name, cfg, args.out, device, dtype, mixed_precision, rank, world_size, args.threshold
            )
        else:  # unlabelled: predictions only
            df = df.reset_index(drop=True)
            df["label"] = -1
            idx, _, probs = predict(model, make_eval_loader(df, cfg, rank, world_size), device, dtype, mixed_precision)
            if rank == 0:
                os.makedirs(args.out, exist_ok=True)
                out = df.iloc[idx].copy()
                out["prob"] = probs
                out["pred"] = (probs >= args.threshold).astype(int)
                out.to_csv(os.path.join(args.out, f"predictions_{name}.csv"), index=False)
    if rank == 0:
        os.makedirs(args.out, exist_ok=True)
        save_json(results, os.path.join(args.out, "metrics.json"))


if __name__ == "__main__":
    main()
