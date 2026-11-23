# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2026 the UroJEPA authors.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Adapted from V-JEPA 2 (src/utils/distributed.py): adds torchrun support and
# returns the local rank so that each process selects its own GPU.

import os

import torch
import torch.distributed as dist

from urojepa.utils.logging import get_logger

logger = get_logger(__name__)


def init_distributed(port=37129):
    """Initialise torch.distributed from torchrun (RANK/WORLD_SIZE/LOCAL_RANK) or SLURM
    (SLURM_PROCID/SLURM_NTASKS/SLURM_LOCALID) environment variables.

    Returns (world_size, rank, local_rank). Falls back to a single process.
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank(), int(os.environ.get("LOCAL_RANK", 0))

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    elif "SLURM_PROCID" in os.environ and int(os.environ.get("SLURM_NTASKS", 1)) > 1:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ.get("SLURM_LOCALID", 0))
        os.environ.setdefault("MASTER_ADDR", os.environ.get("SLURM_LAUNCH_NODE_IPADDR", "localhost"))
        os.environ.setdefault("MASTER_PORT", str(port))
    else:
        return 1, 0, 0

    if world_size == 1:
        return 1, 0, local_rank

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, world_size=world_size, rank=rank)
    logger.info(f"Initialised distributed backend={backend} rank={rank}/{world_size} local_rank={local_rank}")
    return world_size, rank, local_rank


def is_dist():
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def barrier():
    if is_dist():
        dist.barrier()


def all_gather_object(obj):
    """Gather a picklable object from all ranks (returns a list on every rank)."""
    if not is_dist():
        return [obj]
    out = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(out, obj)
    return out


def all_reduce_mean(x: float):
    if not is_dist():
        return x
    t = torch.tensor([x], dtype=torch.float64, device="cuda" if torch.cuda.is_available() else "cpu")
    dist.all_reduce(t)
    return float(t.item() / dist.get_world_size())
