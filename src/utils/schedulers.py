# Copyright 2026 Slyatski Ilya
# Licensed under the Apache License, Version 2.0
# Schedulers: LR, weight decay, EMA tau — from I-JEPA (Assran et al., CVPR 2023)
# EMA tau schedule: momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(total_steps))
# This is the EXACT formula from I-JEPA train.py line ~152

import math


class WarmupCosineSchedule:
    """Linear warmup + cosine annealing LR schedule — from I-JEPA."""

    def __init__(self, optimizer, warmup_steps, start_lr, ref_lr, final_lr, T_max):
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.T_max = T_max - warmup_steps
        self._step = 0

    def step(self):
        self._step += 1
        if self._step < self.warmup_steps:
            progress = float(self._step) / float(max(1, self.warmup_steps))
            new_lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
        else:
            progress = float(self._step - self.warmup_steps) / float(max(1, self.T_max))
            new_lr = max(
                self.final_lr,
                self.final_lr
                + (self.ref_lr - self.final_lr) * 0.5 * (1.0 + math.cos(math.pi * progress)),
            )
        for group in self.optimizer.param_groups:
            group["lr"] = new_lr
        return new_lr


class CosineWDSchedule:
    """Cosine weight decay schedule — from I-JEPA."""

    def __init__(self, optimizer, ref_wd, final_wd, T_max):
        self.optimizer = optimizer
        self.ref_wd = ref_wd
        self.final_wd = final_wd
        self.T_max = T_max
        self._step = 0

    def step(self):
        self._step += 1
        progress = self._step / self.T_max
        new_wd = self.final_wd + (self.ref_wd - self.final_wd) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )
        if self.final_wd <= self.ref_wd:
            new_wd = max(self.final_wd, new_wd)
        else:
            new_wd = min(self.final_wd, new_wd)
        for group in self.optimizer.param_groups:
            if not group.get("WD_exclude", False):
                group["weight_decay"] = new_wd
        return new_wd


class EMATauSchedule:
    """EMA tau schedule — EXACT formula from I-JEPA train.py line ~152:
    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale)
                          for i in range(int(ipe*num_epochs*ipe_scale)+1))

    CRITICAL (user requirement): constant tau is suboptimal.
    - Early training: target should move faster (lower tau)
    - Late training: target should be stable (higher tau)
    """

    def __init__(self, tau_start=0.996, tau_end=0.9999, total_steps=100000):
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.total_steps = total_steps
        self._step = 0

    def step(self):
        self._step += 1
        # I-JEPA formula: ema[0] + i*(ema[1]-ema[0])/total_steps
        i = min(self._step, self.total_steps)
        return self.tau_start + i * (self.tau_end - self.tau_start) / self.total_steps

    def step_cosine(self):
        """Cosine EMA schedule — smoother transition from C-JEPA best practices.

        C-JEPA (NeurIPS 2024 Spotlight) uses cosineA smoother tau transition
        to avoid sudden target encoder freezing. Cosine spends more time
        in the intermediate regime where the target is learning but stable.

        tau(t) = tau_end + (tau_start - tau_end) * 0.5 * (1 + cos(π * t / T))
        """
        self._step += 1
        i = min(self._step, self.total_steps)
        progress = i / self.total_steps
        return self.tau_end + (self.tau_start - self.tau_end) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )
