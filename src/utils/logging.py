# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
# Logging utilities — from I-JEPA (Assran et al., CVPR 2023)
# CSVLogger pattern from src/utils/logging.py in I-JEPA repo


import torch


class AverageMeter:
    """computes and stores the average and current value — from I-JEPA."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.max = float("-inf")
        self.min = float("inf")
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        try:
            self.max = max(val, self.max)
            self.min = min(val, self.min)
        except Exception:
            pass
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CSVLogger:
    """CSV logger — from I-JEPA src/utils/logging.py."""

    def __init__(self, fname, *argv):
        self.fname = fname
        self.types = []
        with open(self.fname, "a+", encoding="utf-8") as f:
            for i, v in enumerate(argv, 1):
                self.types.append(v[0])
                if i < len(argv):
                    print(v[1], end=",", file=f)
                else:
                    print(v[1], end="\n", file=f)

    def log(self, *argv):
        with open(self.fname, "a+", encoding="utf-8") as f:
            for i, tv in enumerate(zip(self.types, argv), 1):
                end = "," if i < len(argv) else "\n"
                print(tv[0] % tv[1], end=end, file=f)


def grad_logger(named_params):
    """Gradient statistics logger — from I-JEPA src/utils/logging.py."""
    stats = AverageMeter()
    stats.first_layer = None
    stats.last_layer = None
    for n, p in named_params:
        if (p.grad is not None) and not (n.endswith(".bias") or len(p.shape) == 1):
            grad_norm = float(torch.norm(p.grad.data))
            stats.update(grad_norm)
            if "qkv" in n:
                stats.last_layer = grad_norm
                if stats.first_layer is None:
                    stats.first_layer = grad_norm
    if stats.first_layer is None or stats.last_layer is None:
        stats.first_layer = stats.last_layer = 0.0
    return stats
