# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Adapted from V-JEPA 2 (src/utils/schedulers.py); LambdaLinearWarmupHold comes from
# V-JEPA 2.1 (app/vjepa_2_1/models/utils/modules.py).

import math


class WarmupCosineSchedule(object):
    """Linear warm-up from start_lr to ref_lr, then cosine decay to final_lr (per step)."""

    def __init__(self, optimizer, warmup_steps, start_lr, ref_lr, T_max, final_lr=0.0):
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.T_max = T_max - warmup_steps
        self._step = 0.0

    def step(self):
        self._step += 1
        if self._step < self.warmup_steps:
            progress = float(self._step) / float(max(1, self.warmup_steps))
            new_lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
        else:
            progress = min(1.0, float(self._step - self.warmup_steps) / float(max(1, self.T_max)))
            new_lr = self.final_lr + (self.ref_lr - self.final_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in self.optimizer.param_groups:
            group["lr"] = new_lr
        return new_lr


class CosineWDSchedule(object):
    """Cosine schedule of weight decay from ref_wd to final_wd (groups with WD_exclude are skipped)."""

    def __init__(self, optimizer, ref_wd, T_max, final_wd=0.0):
        self.optimizer = optimizer
        self.ref_wd = ref_wd
        self.final_wd = final_wd
        self.T_max = T_max
        self._step = 0.0

    def step(self):
        self._step += 1
        progress = min(1.0, self._step / self.T_max)
        new_wd = self.final_wd + (self.ref_wd - self.final_wd) * 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in self.optimizer.param_groups:
            if not group.get("WD_exclude", False):
                group["weight_decay"] = new_wd
        return new_wd


class LambdaLinearWarmupHold:
    """Weight of the V-JEPA 2.1 context (dense) loss: 0 before start_iter, linear
    ramp up to lambda_value at end_iter, constant afterwards."""

    def __init__(self, lambda_value: float, start_iter: int, end_iter: int):
        assert end_iter > start_iter, "end_iter must be > start_iter"
        self.lambda_value = float(lambda_value)
        self.start = int(start_iter)
        self.end = int(end_iter)

    def value(self, global_iter: int) -> float:
        if global_iter < self.start:
            return 0.0
        if global_iter >= self.end:
            return self.lambda_value
        return self.lambda_value * (global_iter - self.start) / (self.end - self.start)
