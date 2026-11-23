# Copyright (c) 2026 the UroJEPA authors.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel


def parse_dtype(name):
    """'bfloat16' | 'float16' | 'float32' -> (torch dtype, use_autocast)"""
    name = (name or "float32").lower()
    if name == "bfloat16":
        return torch.bfloat16, True
    if name == "float16":
        return torch.float16, True
    return torch.float32, False


def make_grad_scaler():
    try:
        return torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):  # torch < 2.3
        return torch.cuda.amp.GradScaler()


def unwrap(module):
    return module.module if isinstance(module, DistributedDataParallel) else module


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
