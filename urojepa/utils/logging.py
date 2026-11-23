# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Adapted from V-JEPA 2 (src/utils/logging.py).

import logging
import sys

import torch

LOG_FORMAT = "[%(levelname)-7s][%(asctime)s][%(name)-24s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name=None, force=False):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT, force=force)
    return logging.getLogger(name=name)


def gpu_timer(closure, log_timings=True):
    """Time `closure()` on the GPU (ms); returns (result, elapsed_ms or -1)."""
    log_timings = log_timings and torch.cuda.is_available()
    elapsed_time = -1.0
    if log_timings:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    result = closure()
    if log_timings:
        end.record()
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end)
    return result, elapsed_time


class CSVLogger(object):

    def __init__(self, fname, *argv, mode="a", delim=","):
        self.fname = fname
        self.types = [v[0] for v in argv]
        self.delim = delim
        with open(self.fname, mode) as f:
            print(delim.join(v[1] for v in argv), file=f)

    def log(self, *argv):
        with open(self.fname, "a") as f:
            print(self.delim.join(t % v for t, v in zip(self.types, argv)), file=f)


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
